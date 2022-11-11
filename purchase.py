import time
import pandas as pd
from tqdm import tqdm
import sklearn.preprocessing as sp
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
# 取出数据源
raw_train = pd.read_csv('../data/train.csv')
raw_test = pd.read_csv('../data/test.csv')
submit_df = pd.read_csv('../data/submit_example.csv')


# 预处理
for df in [raw_train, raw_test]:
    # 处理空值
    for f in ['category_code', 'brand']:
        df[f].fillna('<unkown>', inplace=True)  #用unkown填充Nan，确认修改并覆盖原数据

    # 处理时间
    df['event_time'] = pd.to_datetime(df['event_time'], format='%Y-%m-%d %H:%M:%S UTC')
    df['timestamp'] = df['event_time'].apply(lambda x: time.mktime(x.timetuple()))
    df['timestamp'] = df['timestamp'].astype(int)

# 排序
raw_train = raw_train.sort_values(['user_id', 'timestamp'])
raw_test = raw_test.sort_values(['user_id', 'timestamp'])

# 处理非数值特征
df = pd.concat([raw_train, raw_test], ignore_index=True)

for f in ['event_type', 'category_code', 'brand']:
    # 构建编码器
    le = sp.LabelEncoder()
    le.fit(df[f])

    # 设置新值
    raw_train[f] = le.transform(raw_train[f])
    raw_test[f] = le.transform(raw_test[f])

# 删除无用列
useless = ['event_time', 'user_session', 'timestamp']
for df in [raw_train, raw_test]:
    df.drop(columns=useless, inplace=True)

# 滑动窗口构造数据集
# 为了让机器学习模型能够处理时序数据，必须通过滑动窗口构造数据，后一个时间点的作为前一个时间点的预测值
#
# 训练集数据生成：滑动窗口
# 用前一个时间节点的数据预测后一个时间节点是商品
train_df = pd.DataFrame()
user_ids = raw_train['user_id'].unique()    #按user_id去重，并从大到小排列
for uid in tqdm(user_ids):
    user_data = raw_train[raw_train['user_id'] == uid].copy(deep=True)
    if user_data.shape[0] < 2:
        # 小于两条的，直接忽略
        continue

    user_data['y'] = user_data['product_id'].shift(-1)
    user_data = user_data.head(user_data.shape[0] - 1)
    train_df = train_df.append(user_data)

train_df['y'] = train_df['y'].astype(int)
train_df = train_df.reset_index(drop=True)

# 测试集数据生成，只取每个用户最后一次操作用来做预测
test_df = raw_test.groupby(['user_id'], as_index=False).last()

train_df.drop(columns=['user_id'], inplace=True)

# display(train_df, test_df)

user_ids = test_df['user_id'].unique()

preds = []
for uid in tqdm(user_ids):
    pids = raw_test[raw_test['user_id'] == uid]['product_id'].unique()

    # 找到训练集中有这些product_id的数据作为当前用户的训练集
    p_train = train_df[train_df['product_id'].isin(pids)]

    # 只取最后一条进行预测
    user_test = test_df[test_df['user_id'] == uid].drop(columns=['user_id'])

    X_train = p_train.iloc[:, :-1]
    y_train = p_train['y']

    if len(X_train) > 0:
        # 训练
        clf = xgb.XGBClassifier(**{'seed': int(time.time())})
        clf.fit(X_train, y_train)

        # 预测
        pred = clf.predict(user_test)[0]
    else:
        # 训练集中无对应数据
        # 直接取最后一条数据作为预测值
        pred = user_test['product_id'].iloc[0]

    preds.append(pred)

submit_df['product_id'] = preds


submit_df.to_csv('submit.csv', index=False)
