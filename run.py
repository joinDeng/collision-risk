import kessler
from kessler import EventDataset
from kessler.nn import LSTMPredictor
from kessler.data import kelvins_to_event_dataset
from kessler import cdm
from kessler import util
import pandas as pd

date0 = '2020-05-22T21:41:31.975'  # 定义时间起点
high_risk = -6


# precision:预测为高碰撞概率事件正确个数比例
def Precision(weight_av_cdm, event_test):
    predict_positive = 0
    predict_right = 0
    for i in range(len(weight_av_cdm)):
        if float(weight_av_cdm[i]['COLLISION_PROBABILITY']) > high_risk:
            predict_positive += 1
            if float(event_test[i][-1]['COLLISION_PROBABILITY']) > high_risk:
                predict_right += 1
    if predict_positive == 0:
        predict_positive = 1e-4
    print('precision:{}'.format(predict_right/predict_positive))
    return predict_right/predict_positive


# recall:高碰撞概率事件中预测正确的比例
def Recall(weight_av_cdm, event_test):
    predict_right = 0
    positive = 0
    for i in range(len(weight_av_cdm)):
        if float(event_test[i][-1]['COLLISION_PROBABILITY']) > high_risk:
            positive += 1
            if float(weight_av_cdm[i]['COLLISION_PROBABILITY']) > high_risk:
                predict_right += 1
    if positive == 0:
        positive = 1e-4
    print('recall:{}'.format(predict_right/positive))
    return predict_right/positive


def MSE(weight_av_cdms, event_test):
    num_high_risk = 0
    mse = 0
    for i in range(len(weight_av_cdms)):
        if float(event_test[i][-1]['COLLISION_PROBABILITY']) > high_risk:
            num_high_risk += 1
            difference2 = (float(weight_av_cdms[i]['COLLISION_PROBABILITY']) - float(event_test[i][-1]['COLLISION_PROBABILITY']))**2
            #print('weight_av_cdms[i][\'COLLISION_PROBABILITY\']:', weight_av_cdms[i]['COLLISION_PROBABILITY'])
            #print('event_test[i][-1][\'COLLISION_PROBABILITY\']:', event_test[i][-1]['COLLISION_PROBABILITY'])
            mse += difference2
    print('number of high risk:', num_high_risk)
    print('mse:', mse)
    print('predict MSE:{}'.format(mse/num_high_risk))
    if num_high_risk == 0:
        num_high_risk = 1e-4
    return mse/num_high_risk


def comment(Precision, Recall, MSE_event_test):
    b = 2
    x = ((b**2) * Precision) + Recall
    if x == 0:
        x = 1e-4
    F2 = (1+(b**2)) * Precision * Recall / (x)
    print('F2:', F2)
    if F2 == 0:
        F2 = 1e-4
    L = MSE_event_test / F2
    print('final score:', L)
    return L


def Run(lstm_size, lstm_depth, drop_out, learn_rate):
    # Set the random number generator seed for reproducibility
    Seed = 1
    kessler.seed(Seed)

    # As an example, we first show the case in which the data comes from the Kelvins competition.
    # For this, we built a specific converter that takes care of the conversion from Kelvins format
    # to standard CDM format (the data can be downloaded at https://kelvins.esa.int/collision-avoidance-challenge/data/):
    # events = kelvins_to_event_dataset(file_train_name_, drop_features=['c_rcs_estimate', 't_rcs_estimate'], num_events=2000) #we use only 200 events

    file_name = 'D:\\APP_program\\Pycharm\\project\\download_cdm_data\\download_cdm_2022-4-30\\cdm_from_spacetrack_mod.csv'
    # file_name = r'C:\Users\86166\Desktop\学习资料\大三上\轨道碎片\train_data\train_data.csv'
    events = kelvins_to_event_dataset(file_name, num_events=15000)

    # Descriptive statistics of the event:
    kessler_stats = events.to_dataframe().describe()
    print(kessler_stats)

    # We only use features with numeric content for the training
    # nn_features is a list of the feature names taken into account for the training:
    # it can be edited in case more features want to be added or removed
    nn_features = events.common_features(only_numeric=True)
    print(nn_features)

    # Split data into a test set (10% of the total number of events)
    len_test_set = int(0.1*len(events))
    print('Test data:', len_test_set)
    events_test = events[-len_test_set:]
    print(events_test)

    # The rest of the data will be used for training and validation
    print('Training data:', int(len(events)-len_test_set))
    events_train = events[:len(events)-len_test_set]
    print(events_train)

    # Create an LSTM predictor, specialized to the nn_features we extracted above
    model = LSTMPredictor(
            lstm_size=lstm_size,  # Number of hidden units per LSTM layer #256
            lstm_depth=lstm_depth,  # Number of stacked LSTM layers
            dropout=drop_out,  # Dropout probability
            features=nn_features)  # The list of feature names to use in the LSTM

    # Start training
    model.learn(events_train,
            epochs=10*2,  #4 Number of epochs (one epoch is one full pass through the training dataset)
            lr=learn_rate,  # Learning rate, can decrease it if training diverges
            batch_size=16*2,  # Minibatch size, can be decreased if there are issues with memory use    ##16*4
            device='cuda',  # Can be 'cuda'  if there is a GPU available
            valid_proportion=0.29,  # Proportion（部分） of the data to use as a validation（验证） set internally  ##6:2:2
            num_workers=4,  # Number of multithreaded dataloader workers, 4 is good for performance, but if there are any issues or errors, please try num_workers=1 as this solves issues with PyTorch most of the time
            event_samples_for_stats=4000)  # Number of events to use to compute NN normalization factors, have this number as big as possible (and at least a few thousands)

    # Save the model to a file after training:
    model.save(file_name="LSTM_20epochs_lr10-4_batchsize16.pkl")

    # NN loss plotted to a file:
    model.plot_loss(file_name='plot_loss.pdf')

    '''
    #we show an example CDM from the set:
    events_train_and_val[0][0]

    #we take a single event, we remove the last CDM and try to predict it
    event = events_test[0]
    event_len = len(event)
    print(event)
    event_beginning = event[0:event_len-1]
    print(event_beginning)
    event_evolution = model.predict_event(event_beginning, num_samples=30)  # max_length=14
    '''

    num_samples = 5  ### 采用蒙特卡洛方法（由drop_out产生随机影响），对某一事件产生num_samples个预测事件

    final_cdm_origin, final_weighted_cdm_predict = [], []
    cdms_origin, cdms_predict = [], []
    events_origin, events_predict = [], []

    index_cdm = cdm.CDM()
    weighted_av_cdm = index_cdm
    final_weighted_cdms_predict, weighted_av_cdms = [], []

    features_focus = ['TCA', 'TIME_TO_TCA', 'MISS_DISTANCE', 'COLLISION_PROBABILITY']
    for i_events_test in range(len(events_test)):
        event = events_test[i_events_test]
        event_beginning = event[0:len(event)-1]
        print(event_beginning)

        # 预测cdm用于评价模型；
        cdm_evolution = model.predict(event_beginning)

        predict_step = model.predict_event_step(event_beginning, num_samples=10)
        final_cdm_predict, final_cdms_predict = [], []
        for i in range(len(predict_step)):
            cdm_step = predict_step[i][-1]
            final_cdm_predict.append(cdm_step)  # 取num_samples=10个预测final_cdm
        final_cdms_predict.append(final_cdm_predict)
        final_weighted_cdm_predict = cdm.CDM()
        for i in features_focus:  # 蒙特卡罗多次预测取平均值作为期望
            """
            if i == 'TIME_TO_TCA' or i == 'COLLISION_PROBABILITY' or i == 'MISS_DISTANCE' or i == 'RELATIVE_SPEED':
                sum_final = 0
                for j in range(len(final_cdm_predict)):
                    if final_cdm_predict[j][i] == None:
                        sum_final = None
                        break
                    else:
                        sum_final += float(final_cdm_predict[j][i])
                if sum_final != None:
                    final_weighted_cdm_predict[i] = float(sum_final / len(final_cdm_predict))
                else:
                    final_weighted_cdm_predict[i] = None
            else:
                sum_final = None
                final_weighted_cdm_predict[i] = None
            """

            sum_final = 0
            for j in range(len(final_cdm_predict)):
                if i == 'TCA':  # TCA对应str，需要转化
                    final_cdm_predict[j][i] = util.from_date_str_to_days(final_cdm_predict[j][i])
                if final_cdm_predict[j][i] is None:
                    if i == 'COLLISION_PROBABILITY':
                        sum_final += -6.0
                    elif i == 'MISS_DISTANCE':
                        sum_final += 10000  # miss distance小于 1000m，collision probability大于 -4 为高风险事件
                    else:
                        sum_final = None
                    break
                else:
                    sum_final += float(final_cdm_predict[j][i])
            if sum_final is not None:
                final_weighted_cdm_predict[i] = float(sum_final / len(final_cdm_predict))
                if i == 'TCA':
                    final_weighted_cdm_predict[i] = util.add_days_to_date_str(date0=date0, days=final_weighted_cdm_predict[i])
            else:
                final_weighted_cdm_predict[i] = None
        final_weighted_cdms_predict.append(final_weighted_cdm_predict)

        cdms_origin.append(events_test[i_events_test][-1])

        # 预测事件具有实际意义。采用蒙特卡洛方法（由drop_out产生随机影响），对某一事件产生num_samples个预测事件
        event_evolution = model.predict_event(event_beginning, num_samples)
        events_origin.append(events_test[i_events_test])
        events_predict.append(event_evolution)

        # 量化预测与实际的偏差 距离加权法：权：1/距离**2，加权和：所有 值*对应权 的和， 加权平均：加权和 / 权之和
        # for i in index_cdm.to_dataframe().columns:
        count = 0
        for i in features_focus:
            count += 1
            sum_distance2_reciprocal = 0
            sum_product = 0
            weighted_av_cdm[i] = 0
            if i == 'COLLISION_PROBABILITY':
                for j in range(len(event_evolution)):
                    for k in range(len(event)-1, len(event_evolution[j]), 1):
                        distance = float(event[-1]['TIME_TO_TCA']) - float(event_evolution[j][k]['TIME_TO_TCA'])
                        if distance == 0:
                            distance = 1e-9
                        distance2_reciprocal = 1/(distance**2)
                        sum_distance2_reciprocal += distance2_reciprocal
                        product = float(event_evolution[j][k][i]) * distance2_reciprocal   # 距离加权
                        sum_product += product
                if sum_distance2_reciprocal == 0:
                    weighted_av_cdm[i] = None
                else:
                    weighted_av_cdm[i] = sum_product / sum_distance2_reciprocal
        weighted_av_cdms.append(weighted_av_cdm)   # 对每一个event预测得到的cdm的集合

    with open('cdm_compare_error_percent.txt', "w+") as file_cdm_compare:
        file_cdm_compare.write('compare between the actual cdm and the weighted average predicted cdm\n')
        for i in range(len(cdms_origin)):
            file_cdm_compare.write('origin :')
            for feature in features_focus:
                file_cdm_compare.write(' {}:{} '.format(feature, cdms_origin[i][feature]))
            file_cdm_compare.write('\n')
            file_cdm_compare.write('predict:')
            for feature in features_focus:
                if feature == 'MISS_DISTANCE':
                    file_cdm_compare.write(' {}:{} '.format(feature, str(int(final_weighted_cdms_predict[i][feature]))))
                else:
                    file_cdm_compare.write(' {}:{} '.format(feature, str(final_weighted_cdms_predict[i][feature])))
            file_cdm_compare.write('\n')
            file_cdm_compare.write('error:  ')
            for feature in features_focus:
                if feature == 'MISS_DISTANCE':
                    file_cdm_compare.write(' {}:{:.3f} '.format(feature, int((cdms_origin[i][feature] - final_weighted_cdms_predict[i][feature]))/abs(cdms_origin[i][feature])))
                elif feature == 'TCA':
                    TCA1, TCA2 = util.from_date_str_to_days(cdms_origin[i][feature]), util.from_date_str_to_days(final_weighted_cdms_predict[i][feature])
                    file_cdm_compare.write(' {}:{} '.format(feature, (TCA1 - TCA2)) + ' '*6)
                else:
                    file_cdm_compare.write(' {}:{:.3f} '.format(feature, (cdms_origin[i][feature]-final_weighted_cdms_predict[i][feature])/abs(cdms_origin[i][feature])))
            file_cdm_compare.write('\n')
            file_cdm_compare.write('*'*100 + '\n')
    # 以events_origin[0:len()-1]为依据，进行预测

    with open('event_compare.txt', "w+") as file_event_compare:
        for i in range(len(events_origin)):
            file_event_compare.write('origin event{}:'.format(i))
            for j in range(len(events_origin[i])):
                if j == 0:
                    file_event_compare.write('cdm{}:'.format(j))
                else:
                    file_event_compare.write(' '*14 + 'cdm{}:'.format(j))
                for feature in features_focus:
                    file_event_compare.write(' {}:{}'.format(feature, events_origin[i][j][feature]))
                file_event_compare.write('\n')

            for k in range(num_samples):
                for j in range(len(events_origin[i])-1, len(events_predict[i][k]), 1):
                    if k == 0 and j == len(events_origin[i])-1:
                        file_event_compare.write('predict {}:'.format(i) + ' ' * 3)
                    else:
                        file_event_compare.write(' ' * 14)
                    file_event_compare.write('cdm{}:'.format(j))
                    for feature in features_focus:
                        if feature == 'MISS_DISTANCE':
                            file_event_compare.write(' {}:{}'.format(feature, int(events_predict[i][k][j][feature])))
                        else:
                            file_event_compare.write(' {}:{}'.format(feature, events_predict[i][k][j][feature]))
                    file_event_compare.write('\n')
            file_event_compare.write('*'*100 + '\n')

    pre = Precision(final_weighted_cdms_predict, events_test)
    re = Recall(final_weighted_cdms_predict, events_test)
    mse = MSE(final_weighted_cdms_predict, events_test)
    score = comment(pre, re, mse)

    # 记录最终得分
    file_name_score = r'C:\Users\86166\Desktop\学习资料\大三上\轨道碎片\题目-3 基于神经网络的空间碎片碰撞分析\results\不同参数下的得分和平均相对误差.txt'
    with open(file_name_score, 'a+') as f_score:
        condition = format('lstm_size:{}, lstm_depth:{}, drop_out:{}, learn_rate:{}'.format(lstm_size, lstm_depth, drop_out, learn_rate))
        f_score.write(condition + '  ' + 'final score：%.6f' % score + '\n')


if __name__ == "__main__":
    Run(lstm_size=64, lstm_depth=2, drop_out=0.2, learn_rate=5e-4)

