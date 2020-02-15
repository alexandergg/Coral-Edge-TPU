"""A demo for object detection.

For Raspberry Pi, you need to install 'feh' as image viewer:
sudo apt-get install feh

Example (Running under python-tflite-source/edgetpu directory):

  - Face detection:
    python3.5 demo/object_detection.py \
    --model='test_data/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite' \
    --input='test_data/face.jpg'

  - Pet detection:
    python3.5 demo/object_detection.py \
    --model='test_data/ssd_mobilenet_v1_fine_tuned_edgetpu.tflite' \
    --label='test_data/pet_labels.txt' \
    --input='test_data/pets.jpg'

'--output' is an optional flag to specify file name of output image.
"""
import argparse
import platform
import subprocess
import numpy
import cv2
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from PIL import ImageDraw
from utils.opencv import get_pink_bounding_box

def ReadLabelFile(file_path):
  with open(file_path, 'r') as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = line.strip().split(maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()
  return ret

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='Path of the detection model.', required=True)
  parser.add_argument(
      '--label', help='Path of the labels file.')
  parser.add_argument(
      '--input', help='File path of the input image.', required=True)
  parser.add_argument(
      '--output', help='File path of the output image.')
  args = parser.parse_args()

  if not args.output:
    output_name = 'object_detection_result.jpg'
  else:
    output_name = args.output

  engine = DetectionEngine(args.model)
  labels = ReadLabelFile(args.label) if args.label else None

  img = Image.open(args.input)
  draw = ImageDraw.Draw(img)
  imcv = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)

  ans = engine.DetectWithImage(img, threshold=0.40, keep_aspect_ratio=True,
                               relative_coord=False, top_k=10)
  if ans:
    for obj in ans:
      print ('-----------------------------------------')
      box = obj.bounding_box.flatten().tolist()
      if labels[obj.label_id] == 'white' or labels[obj.label_id] == 'red':
        if get_pink_bounding_box(imcv, box):
          print(labels[5])
        else:
          print(labels[obj.label_id])
      else:
        print(labels[obj.label_id])
      print ('score = ', obj.score)
      print ('box = ', box)
      draw.rectangle(box, outline='red')
    img.save(output_name)
    if platform.machine() == 'x86_64':
      img.show()
    elif platform.machine() == 'armv7l':
      subprocess.Popen(['feh', output_name])
    else:
      print ('Please check ', output_name)
  else:
    print ('No object detected!')

if __name__ == '__main__':
  main()
