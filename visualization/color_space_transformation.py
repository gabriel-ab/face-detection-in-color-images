"""Teste da correção de iluminação"""

from matplotlib import pyplot as plt
from pathlib import Path
from face_detection.color_space_transformation import CB, CR, ChromaNonLinearTransformation

image_path = Path(__file__).parent / 'reference.png'


test_range = list(range(CB.ymin, CB.ymax))

def apply_cb(y_values, cb_value):
    return [CB.transform(y, cb_value) for y in y_values]

def apply_cr(y_values, cr_value):
    return [CR.transform(y, cr_value) for y in y_values]

plt.figure(figsize=(12,6))
plt.subplot(121, title='CB = [80, 110, 140]', xlabel='Y', ylabel='CB', xlim=(0, 300), ylim=(0, 300))
plt.plot(test_range, apply_cb(test_range, 80), label='80')
plt.plot(test_range, apply_cb(test_range, 110), label='110')
plt.plot(test_range, apply_cb(test_range, 140), label='140')
plt.grid()
plt.legend()

plt.subplot(122, title='CR = [130, 145, 155]', xlabel='Y', ylabel='CR', xlim=(0, 300), ylim=(0, 300))
plt.plot(test_range, apply_cr(test_range, 130), label='130')
plt.plot(test_range, apply_cr(test_range, 145), label='145')
plt.plot(test_range, apply_cr(test_range, 155), label='155')
plt.grid()
plt.legend()
plt.show()