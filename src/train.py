from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='custom_data.yaml',
   imgsz=640,
   epochs=100,
   batch=8,
   name='yolov8n_aorta_detection'
)

results = model.val()
success = model.export(format="onnx")