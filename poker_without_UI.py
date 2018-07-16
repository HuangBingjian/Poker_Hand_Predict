'''
作者：黄炳坚

运行poker_with_UI,将显示pyqt制作的UI界面，提供可视化选择操作。
运行poker_without_UI,直接运行程序，选用Poker_Test列表中的值进行测试。
'''

import pandas as pd
import tensorflow as tf

# 用于标注数据集的各列，如1,1,1,13,2,4,2,3,1,12,0
COLUMN_NAMES = ['suit1', 'rank1', 'suit2', 'rank2', 'suit3', 'rank3',
                'suit4', 'rank4', 'suit5', 'rank5', 'poker_hand']

# 数据集标签分类
POKER_HANDS = [ 'Nothing',  'One pair',     'Two pairs',        'Three of a kind',  'Straight',
                'Flush',    'Full house',   'Four of a kind',   'Straight flush',   'Royal flush']

# 用于测试是否能准确识别，用于无界面情况，可任意替换。
Poker_Test = [1,1,2,1,3,9,1,5,2,3]

# 加载数据集的数据并预处理
def load_data(y_name='poker_hand'):
    train_path = './data/poker-hand-training.data'
    test_path = './data/poker-hand-test.data'

    # 对数据集的特征和标签进行分割
    train = pd.read_csv(train_path, names=COLUMN_NAMES, header=None)
    train_x, train_y = train, train.pop(y_name)
    test = pd.read_csv(test_path, names=COLUMN_NAMES, header=None)
    test_x, test_y = test, test.pop(y_name)

    return train_x, train_y, test_x, test_y


def train_input_fn(features, labels, batch_size):
    # 训练集
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size):
    # 评估
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset

def exceptional_case(predict_dict):
    # 对Flush, Straight flush, Royal flush情况进行特殊处理
    flush = []
    straight_flush = []
    royal_flush = []
    for i in range(len(predict_dict['suit1'])):
        predict_set = (predict_dict['rank1'][i],predict_dict['rank2'][i],predict_dict['rank3'][i],
                                                 predict_dict['rank4'][i],predict_dict['rank5'][i],)
        predict_list = list(predict_set)
        # 同花色
        if predict_dict['suit1'][i] == predict_dict['suit2'][i] == predict_dict['suit3'][i] == \
                                        predict_dict['suit4'][i] == predict_dict['suit5'][i]:
            # 数字不重复                                                   # 同花色且数字连续
            if len(predict_list) == 5:
                predict_list.sort()                                        # 对列表内数字进行排序
                if predict_list[0] == 1 and predict_list[1] == 10:        # 同花色且1,10,11,12,13
                    royal_flush.append(i)
                elif predict_list[4]-predict_list[0] == 4:                # 同花色且数字连续
                    straight_flush.append(i)
                else:
                    flush.append(i)
            else:
                flush.append(i)

    return flush, straight_flush, royal_flush


def poker_predict(argv=None):
    # 获取数据
    train_x, train_y, test_x, test_y = load_data()

    # 训练特征
    feature_columns = []
    for key in train_x.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    # DNN
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns, hidden_units=[100,100,100,100,100], n_classes=10)

    # 训练网络
    classifier.train( input_fn=lambda:train_input_fn(train_x, train_y, 100), steps=50000)

    # 评估模型
    eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn(test_x, test_y, 100))

    # 使用测试集来检测模型准确率
    suit1 = []
    rank1 = []
    suit2 = []
    rank2 = []
    suit3 = []
    rank3 = []
    suit4 = []
    rank4 = []
    suit5 = []
    rank5 = []
    expected = []

    for x in range(len(test_x)):
        suit1.append(test_x.values[x][0])
        rank1.append(test_x.values[x][1])
        suit2.append(test_x.values[x][2])
        rank2.append(test_x.values[x][3])
        suit3.append(test_x.values[x][4])
        rank3.append(test_x.values[x][5])
        suit4.append(test_x.values[x][6])
        rank4.append(test_x.values[x][7])
        suit5.append(test_x.values[x][8])
        rank5.append(test_x.values[x][9])
        expected.append(test_y.values[x])

    predict_x = {
        'suit1': suit1, 'rank1': rank1, 'suit2': suit2, 'rank2': rank2,
        'suit3': suit3, 'rank3': rank3, 'suit4': suit4, 'rank4': rank4,
        'suit5': suit5, 'rank5': rank5,  }

    # 使用训练好的分类器进行预测
    predictions = classifier.predict( input_fn=lambda: eval_input_fn(predict_x, labels=None, batch_size=100))

    # Flush, Straight flush, Royal flush 等情况进行特殊处理
    flush, straight_flush, royal_flush = exceptional_case(predict_x)

    count_all = 0                               # 测试集的准确预测数量
    count_kind = [0,0,0,0,0,0,0,0,0,0]          # 各类的数量
    count_right = [0,0,0,0,0,0,0,0,0,0]         # 各类的准确预测数量
    for i, pred_dict, expec in zip(range(len(expected)), predictions, expected):
        if i in flush:
            class_id = 5
            probability = 1
        elif i in straight_flush:
            class_id = 8
            probability = 1
        elif i in royal_flush:
            class_id = 9
            probability = 1
        else:
            class_id = pred_dict['class_ids'][0]
            probability = pred_dict['probabilities'][class_id]

        template = ('\nPrediction is "{}" ({:.3f}%), expected "{}"')
        print(template.format(POKER_HANDS[class_id], 100 * probability, POKER_HANDS[expec]))

        count_kind[expec] += 1
        if class_id == expec:
            count_all += 1
            count_right[class_id] += 1
    count_prop = [float(a)/float(b) for a,b in zip(count_right,count_kind)]

    # 待分类数据
    predict_test = {
        'suit1': [Poker_Test[0],], 'rank1': [Poker_Test[1],], 'suit2': [Poker_Test[2],], 'rank2': [Poker_Test[3],],
        'suit3': [Poker_Test[4],], 'rank3': [Poker_Test[5],], 'suit4': [Poker_Test[6],], 'rank4': [Poker_Test[7],],
        'suit5': [Poker_Test[8],], 'rank5': [Poker_Test[9],], }

    predictions_test = classifier.predict( input_fn=lambda: eval_input_fn(predict_test, labels=None, batch_size=100))

    print('\n-------------------------------------------------------------------')
    print('\n模型评估结果： {accuracy:0.5f}'.format(**eval_result))

    print('\n-------------------------------------------------------------------')
    print('\n测试集的数量： ',len(expected))
    print('\n测试集的正确分类数： ',count_all)
    print('\n测试集的分类准确率： ',float(count_all)/len(expected))

    print('\n-------------------------------------------------------------------')
    print('\n各类的数量： ',count_kind)
    print('\n各类的正确分类数： ',count_right)
    print('\n各类的分类正确率： ',count_prop)

    print('\n-------------------------------------------------------------------')
    print('\n待分类数据： ',Poker_Test)

    for pred in predictions_test:
        class_id = pred['class_ids'][0]
        probability = pred['probabilities'][class_id]
        template = ('\n分类结果： "{}: {}" , 概率为： {:.1f}%')
        print(template.format(class_id, POKER_HANDS[class_id], 100 * probability))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(poker_predict)