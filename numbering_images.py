import os

folder = r'C:\Users\JOEY\Documents\Facial_Recog\dataset\father'
count = 1
for filename in os.listdir(folder):
    extension = os.path.splitext(filename)[1]
    src = os.path.join(folder, filename)
    dst = os.path.join(folder, str(count) + extension)
    os.rename(src, dst)
    count += 1