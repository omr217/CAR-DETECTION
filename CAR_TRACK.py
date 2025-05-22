import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

# Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Class names (adjust according to your YOLO model)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Load the TensorRT engine
def load_engine(trt_file_path):
    with open(trt_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Preprocess input frame
def preprocess_frame(frame, input_shape):
    frame_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    frame_transposed = np.transpose(frame_normalized, (2, 0, 1))  # Convert HWC to CHW
    return np.ascontiguousarray(np.expand_dims(frame_transposed, axis=0))

# Run inference with TensorRT
def infer(context, bindings, inputs, outputs, stream):
    cuda.memcpy_htod_async(inputs['device'], inputs['host'], stream)
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs['host'], outputs['device'], stream)
    stream.synchronize()
    return outputs['host']

# Process detections: draw bounding boxes and class labels
def process_detections(detections, frame, input_shape):
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection[:6]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        label = f"{classNames[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Path to the TensorRT engine
engine_path = "model.trt"
engine = load_engine(engine_path)
context = engine.create_execution_context()

# Get input and output binding names and shapes
input_binding_name = engine.get_tensor_name(0)
output_binding_name = engine.get_tensor_name(1)
input_shape = engine.get_tensor_shape(input_binding_name)
output_shape = engine.get_tensor_shape(output_binding_name)

# Allocate memory
input_size = int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)
input_device = cuda.mem_alloc(input_size)
output_device = cuda.mem_alloc(output_size)
bindings = [int(input_device), int(output_device)]
stream = cuda.Stream()

# Load video source
cap = cv2.VideoCapture("cars.mp4")
if not cap.isOpened():
    print("Failed to open video!")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Preprocess input
    input_host = preprocess_frame(frame, input_shape)
    inputs = {'host': input_host, 'device': input_device}
    outputs = {'host': np.empty(output_shape, dtype=np.float32), 'device': output_device}

    # Run inference
    detections = infer(context, bindings, inputs, outputs, stream)

    # Draw detections on frame
    frame = process_detections(detections, frame, input_shape)

    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Detections", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
