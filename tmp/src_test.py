from src.data_processor import DataProcessor
from src.neural_net import NeuralNet

dp = DataProcessor("honda_sell_data_MODEL.csv")
dp.pre_process()

B = 1
#2.31
hidden = [180, 75, 38, 3]
epoch = 5000

nn = NeuralNet(
    bias=B,
    sum_inp=dp.sum_inp_neu, hidden=hidden,
    samples=dp.train_samples,
    epoch=epoch,
    activ="tanh",
    test_samples=dp.test_samples,
    label_y=dp.model_label_y,
    y_test=dp.y_test
)

nn.train()
nn.test()

