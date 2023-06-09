"""
    由训练过程输出收集而来
"""
import matplotlib.pyplot as plt
from utils import plot_carve

valid_bleu = [0.5477, 0.5644, 0.5870, 0.6072, 0.6267, 0.6411, 0.6516, 0.6589, 0.6652, 0.6704,
              0.6751, 0.6792, 0.6828, 0.6857, 0.6884, 0.6909, 0.6933, 0.6957, 0.6979, 0.6999,
              0.7019, 0.7034, 0.7048, 0.7064, 0.7079, 0.7092, 0.7104, 0.7115, 0.7125, 0.7135,
              0.7143, 0.7150, 0.7159, 0.7166, 0.7173, 0.7180, 0.7186, 0.7191, 0.7197, 0.7202,
              0.7207, 0.7209, 0.7210, 0.7211, 0.7212, 0.7213, 0.7214, 0.7214, 0.7215, 0.7216]

loss = [3.15, 2.28, 2.07, 1.95, 1.87, 1.81, 1.76, 1.73, 1.70, 1.67,
        1.65, 1.63, 1.61, 1.59, 1.58, 1.56, 1.55, 1.54, 1.53, 1.52,
        1.51, 1.50, 1.49, 1.48, 1.48, 1.47, 1.46, 1.46, 1.45, 1.44,
        1.44, 1.43, 1.43, 1.42, 1.42, 1.41, 1.41, 1.40, 1.40, 1.39,
        1.39, 1.38, 1.38, 1.37, 1.37, 1.37, 1.36, 1.36, 1.35, 1.35]

train_lr = [0.01, 0.01, 0.01, 0.00999, 0.00999, 0.0098, 0.0098, 0.0097, 0.0096, 0.0095,
            0.0094, 0.0093, 0.0091, 0.0099, 0.00988, 0.00986, 0.00984, 0.00982, 0.0098, 0.00978,
            0.00976, 0.00973, 0.00970, 0.00968, 0.00965, 0.00962, 0.00959, 0.00956, 0.00952, 0.00949,
            0.00946, 0.00942, 0.00938, 0.00934, 0.0093, 0.00926, 0.00922, 0.00918, 0.00914, 0.00909,
            0.00905, 0.009, 0.00895, 0.0089, 0.0085, 0.0088, 0.00875, 0.0087, 0.00864, 0.00859]

plot_carve(title="valid_bleu", save_path='../res_img/valid_bleu.png', x=50, y=valid_bleu)
plot_carve(title='train_loss', save_path='../res_img/train_loss.png', x=50, y=loss)
plot_carve(title='train_lr', save_path='../res_img/train_lr.png', x=50, y=train_lr)
