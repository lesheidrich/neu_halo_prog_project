from src.data_processor import DataProcessor
from src.neural_net import NeuralNet

if __name__ == '__main__':
    dp = DataProcessor("honda_sell_data_MODEL.csv")
    # dp = DataProcessor("honda_sell_data_OG.csv")
    dp.pre_process()

    # hidden = [180, 50, 8]     # 2.4 : drops better at first but converges slowly after 4 to a halt around
    # hidden = [180, 75, 8]     # 2.3 : starts off a bit better and turns log later
    # hidden = [180, 45, 8]     # 2.46
    # hidden = [180, 80, 35]    # 3.65
    # hidden = [180, 80, 5]     # 2.35
    hidden = [180, 75, 38, 3]  # 2.31
    B = 1
    epoch = 5000
    act_f = "sig"
    # act_f = "tanh"
    learn_rate = 0.6

    nn = NeuralNet(bias=B, sum_inp=dp.sum_inp_neu, hidden=hidden, samples=dp.train_samples, epoch=epoch, activ=act_f,
                   test_samples=dp.test_samples, label_y=dp.model_label_y, y_test=dp.y_test, learn_rate=learn_rate)

    nn.train()
    nn.test()
