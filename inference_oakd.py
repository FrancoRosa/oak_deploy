import cv2
import depthai as dai
import numpy as np
import time


text_color = (0, 0, 255)
bbox_color = (0, 255, 0)
# nnBlobPath = "weights_blob/best_openvino_2022.1_6shave.blob"
nnBlobPath = "best_openvino_trees_2022.1_5shave.blob"

labelMap = ['crate', 'junction box', 'module',
            'panel', 'person', 'pile', 'pipe', 'pipes', 'tube']

syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)

yolo_spatial_det_nn = pipeline.createYoloSpatialDetectionNetwork()
yolo_spatial_det_nn.setConfidenceThreshold(0.5)
yolo_spatial_det_nn.setBlobPath(nnBlobPath)
yolo_spatial_det_nn.setNumClasses(9)  # Adjust based on your model
yolo_spatial_det_nn.setCoordinateSize(4)
yolo_spatial_det_nn.setAnchors(
    [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])  # Adjust based on your model
yolo_spatial_det_nn.setAnchorMasks(
    {"side26": [1, 2, 3], "side13": [3, 4, 5]})  # Adjust based on your model
yolo_spatial_det_nn.setIouThreshold(0.5)
yolo_spatial_det_nn.setDepthLowerThreshold(100)
yolo_spatial_det_nn.setDepthUpperThreshold(5000)


monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")

# Properties
camRgb.setPreviewSize(640, 640)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# Setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth map to the perspective of RGB camera, on which inference is done
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setSubpixel(True)
stereo.setOutputSize(monoLeft.getResolutionWidth(),
                     monoLeft.getResolutionHeight())


# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(yolo_spatial_det_nn.input)
if syncNN:
    yolo_spatial_det_nn.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

yolo_spatial_det_nn.out.link(xoutNN.input)

stereo.depth.link(yolo_spatial_det_nn.inputDepth)
yolo_spatial_det_nn.passthroughDepth.link(xoutDepth.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(
        name="detections", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)

    while True:
        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()
        depth = depthQueue.get()

        counter += 1
        current_time = time.monotonic()
        if (current_time - startTime) > 1:
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        frame = inPreview.getCvFrame()

        depthFrame = depth.getFrame()  # depthFrame values are in millimeters

        depth_downscaled = depthFrame[::4]
        if np.all(depth_downscaled == 0):
            min_depth = 0  # Set a default minimum depth value when all elements are zero
        else:
            min_depth = np.percentile(
                depth_downscaled[depth_downscaled != 0], 1)
        max_depth = np.percentile(depth_downscaled, 99)
        depthFrameColor = np.interp(
            depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        detections = inDet.detections

        # If the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width = frame.shape[1]
        for detection in detections:
            roiData = detection.boundingBoxMapping
            roi = roiData.roi
            roi = roi.denormalize(
                depthFrameColor.shape[1], depthFrameColor.shape[0])
            topLeft = roi.topLeft()
            bottomRight = roi.bottomRight()
            xmin = int(topLeft.x)
            ymin = int(topLeft.y)
            xmax = int(bottomRight.x)
            ymax = int(bottomRight.y)

            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            try:
                label = labelMap[detection.label]
            except:
                label = detection.label
            cv2.putText(frame, str(label), (x1 + 10, y1 + 20),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, text_color)
            cv2.putText(frame, "{:.2f}".format(detection.confidence*100),
                        (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, text_color)
            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (
                x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, text_color)
            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (
                x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, text_color)
            cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (
                x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, text_color)

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          bbox_color, cv2.FONT_HERSHEY_SIMPLEX)

        cv2.putText(frame, "NN fps: {:.2f}".format(
            fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
        cv2.imshow("preview", frame)

        if cv2.waitKey(1) == ord('q'):
            break
