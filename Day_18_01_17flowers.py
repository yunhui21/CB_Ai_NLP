# # Day_18_01_17flowers.py
import Image
import os, sys

# def resizeImage(infile, output_dir="data/17flowers_origin", size=(1024,768)):
#      outfile = os.path.splitext(infile)[0]+"_resized"
#      extension = os.path.splitext(infile)[1]
#
#      if (cmp(extension, ".jpg")):
#         return
#
#      if infile != outfile:
#         try :
#             im = Image.open(infile)
#             im.thumbnail(size, Image.ANTIALIAS)
#             im.save(output_dir+outfile+extension,"JPEG")
#         except IOError:
#             print "cannot reduce image for ", infile
#
#
# if __name__=="__main__":
#     output_dir = "resized"
#     dir = os.getcwd()
#
#     if not os.path.exists(os.path.join(dir,output_dir)):
#         os.mkdir(output_dir)
#
#     for file in os.listdir(dir):
#         resizeImage(file,output_dir)

# import Image
# import os
# import sys
#
# directory = sys.argv[1]
#
# for file_name in os.listdir(directory):
#   print("Processing %s" % file_name)
#   image = Image.open(os.path.join(directory, file_name))
#
#   # x,y = image.size
#   new_dimensions = (200, 200)
#   output = image.resize(new_dimensions, Image.ANTIALIAS)
#
#   output_file_name = os.path.join(directory, "small_" + file_name)
#   output.save(output_file_name, "JPEG", quality = 95)
#
# print("All done")

import os
from PIL import Image

resize_method = Image.ANTIALIAS
# Image.NEAREST)  # use nearest neighbour
# Image.BILINEAR) # linear interpolation in a 2x2 environment
# Image.BICUBIC) # cubic spline interpolation in a 4x4 environment
# Image.ANTIALIAS) # best down-sizing filter
# import os
# from PIL import Image
# resize_method = Image.ANTIALIAS
#
# max_height = 1200
# max_width = 1200
# extensions = ['JPG']
#
# path = os.path.abspath(".")
#
#
# def adjusted_size(width, height):
#     if width > max_width or height > max_height:
#         if width > height:
#             return max_width, int(max_width * height / width)
#         else:
#             return int(max_height * width / height), max_height
#     else:
#         return width, height
#
#
# if __name__ == "__main__":
#     for f in os.listdir(path):
#         if os.path.isfile(os.path.join(path, f)):
#             f_text, f_ext = os.path.splitext(f)
#             f_ext = f_ext[1:].upper()
#             if f_ext in extensions:
#                 print
#                 f
#                 image = Image.open(os.path.join(path, f))
#                 width, height = image.size
#                 image = image.resize(adjusted_size(width, height))
#                 image.save(os.path.join(path, f))

import cv2
import glob
import os

inputFolder = 'data/17flowers_orign'
folderlen = len(inputFolder)
os.mkdir('17flowers_224')
i = 0
for img in glob.glob(inputFolder + "/*.jpg")
    image = cv2.imread(img)
    imgResized = cv2.resize(image, (224, 224))
    cv2.imwrite("17flowers_224"+ img[folderlen:], imgResized)
    i += 1
    cv2.imahow('image', imgResized)
    cv2.waitKey(30)
cv2.destroyAllWindows()
