from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import datetime
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
import seaborn as sns

# required machine learning packages
from sklearn import model_selection
from sklearn.feature_selection import RFE
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.calibration import CalibratedClassifierCV as CCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
import xgboost as xgb

pd.set_option("display.max_columns", None)  # Show all columns
# Prevent wrapping to next line
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.width", None)  # Auto-detect the display width

df = pd.read_csv('/Users/surabhisuman/Downloads/spreadspoke_scores.csv')
teams = pd.read_csv('/Users/surabhisuman/Downloads/nfl_teams.csv')
games_elo = pd.read_csv('/Users/surabhisuman/Downloads/nfl-elo 2/nfl_elo.csv')
games_elo_22 = pd.read_csv(
    '/Users/surabhisuman/Downloads/nfl-elo 2/nfl_elo_latest.csv')
games_elo = games_elo.append(games_elo_22)

# removing rows with null values
df = df.replace(r'^\s*$', np.nan, regex=True)
print(df.columns)

# removing rows from specific columns that have null values, resetting index and changing data types
df = df[(df['score_home'].isnull() == False)]
df = df[(df['team_favorite_id'].isnull() == False)]
df = df[(df['over_under_line'].isnull() == False)]
df = df[(df['schedule_season'] >= 1979)]
df.reset_index(drop=True, inplace=True)
df['over_under_line'] = df['over_under_line'].astype(float)

# mapping team_id to the correct teams
df['team_home'] = df['team_home'].map(
    teams.set_index('team_name')['team_id'].to_dict())
df['team_away'] = df['team_away'].map(
    teams.set_index('team_name')['team_id'].to_dict())
print(df)

# Create separate columns for favorite home and away
for i, row in df.iterrows():
    if row['team_favorite_id'] == row['team_home']:
        df.at[i, 'fav_home'] = row['team_home']
    else:
        df.at[i, 'fav_away'] = row['team_away']

# replace na with zeroes in 'fav_home' and 'fav_away'
df['fav_home'].fillna(0, inplace=True)
df['fav_away'].fillna(0, inplace=True)

# create over column
for i, row in df.iterrows():
    if (row['score_home']+row['score_away'] > row['over_under_line']):
        df.at[i, 'over'] = 1

 # replace na with zeroes in 'over'
df['over'].fillna(0, inplace=True)

# recode boolean columns to integer values
df['schedule_playoff'] = df['schedule_playoff'].astype(int)
df['stadium_neutral'] = df['stadium_neutral'].astype(int)

# change data type of date columns in df
df['schedule_date'] = pd.to_datetime(df['schedule_date'])

# correcting schedule week column errors
df.loc[(df['schedule_week'] == '18'), 'schedule_week'] = '17'
df.loc[(df['schedule_week'] == 'Wildcard') | (
    df['schedule_week'] == 'WildCard'), 'schedule_week'] = '18'
df.loc[(df['schedule_week'] == 'Division'), 'schedule_week'] = '19'

df.loc[(df['schedule_week'] == 'Conference'), 'schedule_week'] = '20'
df.loc[(df['schedule_week'] == 'Superbowl') | (
    df['schedule_week'] == 'SuperBowl'), 'schedule_week'] = '21'
df['schedule_week'] = df['schedule_week'].astype(int)

# drop columns not necessary for analysis from df
df = df.drop(columns=['schedule_playoff',
             'weather_humidity', 'weather_detail'], axis=1)


df.to_csv('/Users/surabhisuman/Downloads/spreadspoke_scores_new.csv')
print(df.isna().sum())

# cleaning games_elo data to append with df
print(games_elo)
print(games_elo.keys())
print(games_elo.describe())
print(games_elo.dtypes)

games_elo.to_csv('/Users/surabhisuman/Downloads/games_elo.csv')

# if team_id in df and games_elo are the same
print(set(df['team_away']).intersection(set(games_elo['team1'])))
print(set(df['team_away']).intersection(set(games_elo['team2'])))
print(set(df['team_home']).intersection(set(games_elo['team1'])))
print(set(df['team_home']).intersection(set(games_elo['team2'])))

print(df['team_away'].isin(games_elo['team1']).value_counts())

# Clean up team names
wsh_map = {'WSH': 'WAS'}
games_elo.loc[games_elo.team1 == 'WSH', 'team1'] = 'WAS'
games_elo.loc[games_elo.team2 == 'WSH', 'team2'] = 'WAS'

df.loc[(df.schedule_date == '2016-09-19') & (df.team_home == 'MIN'),
       'schedule_date'] = datetime.datetime(2016, 9, 18)
df.loc[(df.schedule_date == '2017-01-22') & (df.schedule_week == 21),
       'schedule_date'] = datetime.datetime(2017, 2, 5)
df.loc[(df.schedule_date == '1990-01-27') & (df.schedule_week == 21),
       'schedule_date'] = datetime.datetime(1990, 1, 28)
df.loc[(df.schedule_date == '1990-01-13'),
       'schedule_date'] = datetime.datetime(1990, 1, 14)
games_elo.loc[(games_elo.date == '2016-01-09'),
              'date'] = datetime.datetime(2016, 1, 10)
games_elo.loc[(games_elo.date == '2016-01-08'),
              'date'] = datetime.datetime(2016, 1, 9)
games_elo.loc[(games_elo.date == '2016-01-16'),
              'date'] = datetime.datetime(2016, 1, 17)
games_elo.loc[(games_elo.date == '2016-01-15'),
              'date'] = datetime.datetime(2016, 1, 16)

games_elo['date'] = pd.to_datetime(games_elo['date'])

df = df.merge(games_elo, left_on=['schedule_date', 'team_home', 'team_away'], right_on=[
              'date', 'team1', 'team2'], how='left')
games_elo2 = games_elo.rename(
    columns={'team1': 'team2', 'team2': 'team1', 'elo1': 'elo2', 'elo2': 'elo1'})
df.to_csv('/Users/surabhisuman/Downloads/merged.csv')

df = df.merge(games_elo2, left_on=['schedule_date', 'team_home', 'team_away'], right_on=[
              'date', 'team1', 'team2'], how='left')
# df.to_csv('/Users/surabhisuman/Downloads/games_elo_2_new.csv')
print(df.keys())

# removing all _y columns from the dataframe
df = df.drop(df.loc[:, 'playoff_y':'total_rating_y'].columns, axis=1)
print(df.keys())
print(df.shape)

# print(df.isnull().sum())

# drop columns that have more than 70% null values
df = df.drop(df.loc[:, 'importance_x':'neutral_y'].columns, axis=1)
df = df.drop(columns=['playoff_x'])


# remove _x ending from column names
df.columns = df.columns.str.replace('_x', '')
df['result'] = (df['score_home'] > df['score_away'])

# convert boolean column 'result' into integer
df['result'] = df['result'].astype(int)
df.to_csv('/Users/surabhisuman/Downloads/finaldataset.csv')

# Data exploration
# select columns that contribute to analysis
df = df[['schedule_date', 'schedule_season', 'schedule_week', 'team_home',
         'score_home', 'score_away', 'team_away', 'team_favorite_id',
         'spread_favorite', 'over_under_line', 'stadium', 'stadium_neutral',
         'weather_temperature', 'weather_wind_mph', 'fav_home', 'fav_away',
         'over', 'date', 'season', 'neutral', 'team1', 'team2', 'elo1_pre',
         'elo2_pre', 'elo_prob1', 'elo_prob2', 'elo1_post', 'elo2_post', 'result']]

print("columns: ", df.columns)
df.to_csv('/Users/surabhisuman/Downloads/finaldataset1.csv')
# check null values by columns
print(df.isnull().sum())

# summary statistics
print(df.describe().transpose())


# plot correlation
matrix = df.corr().round(2)
# print(matrix)

f, ax = plt.subplots(figsize=(12, 10))

# Generate a mask for upper traingle
mask = np.triu(np.ones_like(matrix, dtype=bool))

# Configure a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap
sns.heatmap(matrix, annot=True, mask=mask, cmap=cmap)
plt.show()

# some percentages to take into consideration when betting

# Calculate home and away win percentage when the games were not played in a neutral stadium
home_win = "{:.2f}".format(
    (sum((df['result'] == 1) & (df['stadium_neutral'] == 0)) / len(df)) * 100)
away_win = "{:.2f}".format(
    (sum((df['result'] == 0) & (df['stadium_neutral'] == 0)) / len(df)) * 100)

# Calculate Under and Over percentage
under_line = "{:.2f}".format(
    (sum((df['score_home'] + df['score_away']) < df['over_under_line']) / len(df)) * 100)
over_line = "{:.2f}".format(
    (sum((df['score_home'] + df['score_away']) > df['over_under_line']) / len(df)) * 100)

# Calculate favoured win percentage for either home team or away team
# Step 1:- replace any string in fav_home and fave_away with 1

df['fav_home'] = df['fav_home'].replace(to_replace='.*', value=1, regex=True)
df['fav_away'] = df['fav_away'].replace(to_replace='.*', value=1, regex=True)

favored = "{:.2f}".format((sum(((df['fav_home'] == 1) & (df['result'] == 1)) | ((df['fav_away'] == 1) & (df['result'] == 0)))
                           / len(df)) * 100)

# Calculate Cover The Spread Percentage & Against The Spread Percentage
cover = "{:.2f}".format((sum(((df['fav_home'] == 1) & ((df['score_away']) - df['score_home'] < df['spread_favorite'])) |
                             ((df['fav_away'] == 1) & ((df['score_away'] - df['score_home']) < df['spread_favorite'])))
                         / len(df)) * 100)

ats = "{:.2f}".format((sum(((df['fav_home'] == 1) & ((df['score_away']) - df['score_home'] > df['spread_favorite'])) |
                           ((df['fav_away'] == 1) & ((df['score_away']) - df['score_home']) > df['spread_favorite']))
                       / len(df)) * 100)


print("Number of Games: " + str(len(df)))
print("Home Straight Up Win Percentage: " + home_win + "%")
print("Away Straight Up Win Percentage: " + away_win + "%")
print("Under Percentage: " + under_line + "%")
print("Over Percentage: " + over_line + "%")
print("Favored Win Percentage: " + favored + "%")
print("Cover The Spread Percentage: " + cover + "%")
print("Against The Spread Percentage: " + ats + "%")

# creating 2 separate dataframes with the home teams / scores and the away teams / scores
# Calculate mean of home score and away score grouped by schedule_season, sechedule_week, team_home and team_away

score = df.groupby(['schedule_season', 'schedule_week', 'team_home']).mean()[
    ['score_home', 'score_away']].reset_index()
aw_score = df.groupby(['schedule_season', 'schedule_week', 'team_away']).mean()[
    ['score_home', 'score_away']].reset_index()

# create total pts column
score['point_diff'] = score['score_home'] - score['score_away']
aw_score['point_diff'] = aw_score['score_away'] - aw_score['score_home']

# append the two dataframes
score = pd.concat([score, aw_score], ignore_index=True)

# fill null values
score['team_home'].fillna(score['team_away'], inplace=True)

# print(score)
# sort by season and week
score.sort_values(['schedule_season', 'schedule_week'],
                  ascending=[True, True], inplace=True)

# removing unneeded columns & changing column name
score = score[['schedule_season', 'schedule_week', 'team_home', 'point_diff']]
score.rename(columns={'team_home': 'team'}, inplace=True)

score.to_csv('/Users/surabhisuman/Downloads/score.csv')

# dictionary of dataframes - separate dataframe for each team
tm_dict = {}
for key in score['team'].unique():
    tm_dict[key] = score[score['team'] == key].reset_index(drop=True)

# dataframe to populate
pts_diff = pd.DataFrame()

# for loop to create a moving average of the previous games for each season
for yr in score['schedule_season'].unique():
    for tm in score['team'].unique():
        data = tm_dict[tm].copy()
        data = data[data['schedule_season'] == yr]

        data.loc[:, 'avg_pts_diff'] = data['point_diff'].shift().expanding().mean()

        pts_diff = pd.concat([pts_diff, data], ignore_index=True)

# merging to df and changing column names
df = df.merge(pts_diff[['schedule_season', 'schedule_week', 'team', 'avg_pts_diff']],
              left_on=['schedule_season', 'schedule_week', 'team_home'], right_on=['schedule_season', 'schedule_week', 'team'],
              how='left')

df.rename(columns={'avg_pts_diff': 'hm_avg_pts_diff'}, inplace=True)

df = df.merge(pts_diff[['schedule_season', 'schedule_week', 'team', 'avg_pts_diff']],
              left_on=['schedule_season', 'schedule_week', 'team_away'], right_on=['schedule_season', 'schedule_week', 'team'],
              how='left')


df.rename(columns={'avg_pts_diff': 'aw_avg_pts_diff'}, inplace=True)

# average point differential over entire season
total_season = pts_diff.groupby(['schedule_season', 'team']).mean()[
    'point_diff'].reset_index()

# adding schedule week for merge and adding one to the season for predictions
total_season['schedule_week'] = 1
total_season['schedule_season'] += 1

# cleaning of columns
df = df[['schedule_date', 'schedule_season', 'schedule_week', 'team_home',
         'team_away', 'team_favorite_id', 'spread_favorite', 'over_under_line',
         'weather_temperature', 'weather_wind_mph', 'score_home', 'score_away', 'stadium_neutral', 'fav_home',
         'fav_away', 'hm_avg_pts_diff', 'aw_avg_pts_diff', 'elo1_pre',
         'elo2_pre', 'elo_prob1', 'elo_prob2', 'elo1_post', 'elo2_post', 'result']]

# removing all rows with null values
df = df.dropna(how='any', axis=0)

print(df.keys())

# Feature Selection
# initial features possible for model for home favourite
X = df[['schedule_season', 'schedule_week', 'over_under_line', 'spread_favorite', 'weather_temperature', 'weather_wind_mph',
        'fav_home', 'hm_avg_pts_diff', 'aw_avg_pts_diff', 'elo_prob1', 'elo_prob2'
        ]]

y = df['result']

#  base model
base = LDA()  # dimensionality reduction

# choose 5 best features
# Reccursive feature elemination
rfe = RFE(estimator=base, n_features_to_select=5)
rfe = rfe.fit(X, y)

# get the selected feature indices
feature_indices = rfe.support_

# get the feature names using the column names of the X dataframe
feature_names = X.columns[feature_indices]

print("Selected features:", feature_names)

# best 5 features chosen by the RFE base model
final_x = df[['spread_favorite', 'fav_home', 'hm_avg_pts_diff', 'elo_prob1',
              'elo_prob2']]

# prepare models
models = []

models.append(('LRG', LogisticRegression(solver='liblinear')))
models.append(('KNB', KNeighborsClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('XGB', xgb.XGBClassifier(random_state=0)))
models.append(('RFC', RandomForestClassifier(
    random_state=0, n_estimators=100)))
models.append(('DTC', DecisionTreeClassifier(
    random_state=0, criterion='entropy', max_depth=5)))

# evaluate each model by average and standard deviations of roc auc
results = []
names = []

# Select the best model based on high roc auc score with low standard deviation

for name, m in models:
    kfold = model_selection.KFold(n_splits=5, random_state=1, shuffle=True)
    cv_results = model_selection.cross_val_score(
        m, final_x, y, cv=kfold, scoring='roc_auc')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# training and testing data (2021 and 2022)

train = df.copy()
test = df.copy()

train = train.loc[train['schedule_season'] < 2022]
test = test.loc[test['schedule_season'] > 2021]
X_train = train[['over_under_line', 'spread_favorite',
                 'fav_home', 'hm_avg_pts_diff', 'elo_prob1']]
y_train = train['result']
X_test = test[['over_under_line', 'spread_favorite',
               'fav_home', 'hm_avg_pts_diff', 'elo_prob1']]
y_test = test['result']

# calibrate probabilities and fit model to training data
boost = xgb.XGBClassifier()
dtc = DecisionTreeClassifier(max_depth=5, criterion='entropy')
lrg = LogisticRegression(solver='liblinear')
vote = VotingClassifier(
    estimators=[('boost', boost), ('dtc', dtc), ('lrg', lrg)], voting='soft')

model = CCV(vote, method='isotonic', cv=3)
model.fit(X_train, y_train)
# vote.fit(X_train, y_train)
# predict probabilities
predicted = model.predict_proba(X_test)[:, 1]

# ROC AUC Score higher is better while Brier Score the lower the better
print("Metrics" + "\t\t" + "My Model" + "\t" + "Elo Results")
print("ROC AUC Score: " + "\t" + "{:.4f}".format(roc_auc_score(y_test, predicted)
                                                 ) + "\t\t" + "{:.4f}".format(roc_auc_score(test['result'], test['elo_prob1'])))
print("Brier Score: " + "\t" + "{:.4f}".format(brier_score_loss(y_test, predicted)) +
      "\t\t" + "{:.4f}".format(brier_score_loss(test['result'], test['elo_prob1'])))

# creating a column with the models probabilities to analyze vs elo fivethirtyeight
test.loc[:, 'hm_prob'] = predicted
test = test[['schedule_season', 'schedule_week', 'team_home',
             'team_away', 'elo_prob1', 'hm_prob', 'result']]

# calulate bets won (only make a bet when probability is greater than / equal to 60% or less than / equal to 40%)
test['my_bet_won'] = (((test['hm_prob'] >= 0.60) & (test['result'] == 1)) | (
    (test['hm_prob'] <= 0.40) & (test['result'] == 0))).astype(int)
test['elo_bet_won'] = (((test['elo_prob1'] >= 0.60) & (test['result'] == 1)) | (
    (test['elo_prob1'] <= 0.40) & (test['result'] == 0))).astype(int)

# calulate bets lost (only make a bet when probability is greater than / equal to 60% or less than / equal to 40%)
test['my_bet_lost'] = (((test['hm_prob'] >= 0.60) & (test['result'] == 0)) | (
    (test['hm_prob'] <= 0.40) & (test['result'] == 1))).astype(int)
test['elo_bet_lost'] = (((test['elo_prob1'] >= 0.60) & (test['result'] == 0)) | (
    (test['elo_prob1'] <= 0.40) & (test['result'] == 1))).astype(int)

# printing some quick overall results for my model
print("My Model Win Percentage: " + "{:.4f}".format(test['my_bet_won'].sum() / (
    test['my_bet_lost'].sum() + test['my_bet_won'].sum())))
print("Total Number of Bets Won: " + str(test['my_bet_won'].sum()))
print("Total Number of Bets Made: " +
      str((test['my_bet_lost'].sum() + test['my_bet_won'].sum())))
print("Possible Games: " + str(len(test)))

# printing some quick overall results for fivethirtyeight's ELO model
print("ELO Model Win Percentage: " + "{:.4f}".format(test['elo_bet_won'].sum()/(
    test['elo_bet_lost'].sum() + test['elo_bet_won'].sum())))
print("Total Number of Bets Won: " + str(test['elo_bet_won'].sum()))
print("Total Number of Bets Made: " +
      str((test['elo_bet_lost'].sum() + test['elo_bet_won'].sum())))
print("Possible Games: " + str(len(test)))


# Hyperparameter Tuning using GridSearchCV

# Define the parameter grid for the hyperparameters to tune
param_grid = {
    'boost__learning_rate': [0.1, 0.01],
    'dtc__max_depth': [3, 5, 7],
    'lrg__C': [0.1, 1, 10]
}
# Create the GridSearchCV object
grid = GridSearchCV(vote, param_grid, cv=3)

# Fit the GridSearchCV object to the training data

grid.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding score
print('Best hyperparameters:', grid.best_params_)
print('Best score:', grid.best_score_)

# predict probabilities
# calibrate probabilities and fit model to training data
boost = xgb.XGBClassifier(learning_rate=0.1)
dtc = DecisionTreeClassifier(max_depth=3, criterion='entropy')
lrg = LogisticRegression(solver='liblinear', C=1)
vote = VotingClassifier(
    estimators=[('boost', boost), ('dtc', dtc), ('lrg', lrg)], voting='soft')

model = CCV(vote, method='isotonic', cv=3)
model.fit(X_train, y_train)
# vote.fit(X_train, y_train)
# predict probabilities
predicted_hyp = model.predict_proba(X_test)[:, 1]

# ROC AUC Score higher is better while Brier Score the lower the better
print('**********Hyperparameter Tuned Model**********')
print("Metrics" + "\t\t" + "My Model" + "\t" + "Elo Results")
print("ROC AUC Score: " + "\t" + "{:.4f}".format(roc_auc_score(y_test, predicted_hyp)
                                                 ) + "\t\t" + "{:.4f}".format(roc_auc_score(test['result'], test['elo_prob1'])))
print("Brier Score: " + "\t" + "{:.4f}".format(brier_score_loss(y_test, predicted_hyp)) +
      "\t\t" + "{:.4f}".format(brier_score_loss(test['result'], test['elo_prob1'])))

# creating a column with the models probabilities to analyze vs elo fivethirtyeight
test.loc[:, 'hm_prob'] = predicted_hyp
test = test[['schedule_season', 'schedule_week', 'team_home',
             'team_away', 'elo_prob1', 'hm_prob', 'result']]

# calulate bets won (only make a bet when probability is greater than / equal to 60% or less than / equal to 40%)
test['my_bet_won'] = (((test['hm_prob'] >= 0.60) & (test['result'] == 1)) | (
    (test['hm_prob'] <= 0.40) & (test['result'] == 0))).astype(int)
test['elo_bet_won'] = (((test['elo_prob1'] >= 0.60) & (test['result'] == 1)) | (
    (test['elo_prob1'] <= 0.40) & (test['result'] == 0))).astype(int)

# calulate bets lost (only make a bet when probability is greater than / equal to 60% or less than / equal to 40%)
test['my_bet_lost'] = (((test['hm_prob'] >= 0.60) & (test['result'] == 0)) | (
    (test['hm_prob'] <= 0.40) & (test['result'] == 1))).astype(int)
test['elo_bet_lost'] = (((test['elo_prob1'] >= 0.60) & (test['result'] == 0)) | (
    (test['elo_prob1'] <= 0.40) & (test['result'] == 1))).astype(int)

# printing some quick overall results for my model
print("My Model Win Percentage: " + "{:.4f}".format(test['my_bet_won'].sum() / (
    test['my_bet_lost'].sum() + test['my_bet_won'].sum())))
print("Total Number of Bets Won: " + str(test['my_bet_won'].sum()))
print("Total Number of Bets Made: " +
      str((test['my_bet_lost'].sum() + test['my_bet_won'].sum())))
print("Possible Games: " + str(len(test)))

# printing some quick overall results for fivethirtyeight's ELO model
print("ELO Model Win Percentage: " + "{:.4f}".format(test['elo_bet_won'].sum()/(
    test['elo_bet_lost'].sum() + test['elo_bet_won'].sum())))
print("Total Number of Bets Won: " + str(test['elo_bet_won'].sum()))
print("Total Number of Bets Made: " +
      str((test['elo_bet_lost'].sum() + test['elo_bet_won'].sum())))
print("Possible Games: " + str(len(test)))


# Testing the model using Hard Voting Ensemble Method
# calibrate probabilities and fit model to training data
boost = xgb.XGBClassifier()
dtc = DecisionTreeClassifier(max_depth=5, criterion='entropy')
lrg = LogisticRegression(solver='liblinear')
vote = VotingClassifier(
    estimators=[('boost', boost), ('dtc', dtc), ('lrg', lrg)], voting='hard')

X_test_array = X_test.to_numpy()
vote.fit(X_train, y_train)
# predict probabilities
predicted_hd = vote.predict(X_test_array)

# ROC AUC Score higher is better while Brier Score the lower the better
print('**********Hard Voting Ensemble Model**********')
print("Metrics" + "\t\t" + "My Model" + "\t" + "Elo Results")
print("ROC AUC Score: " + "\t" + "{:.4f}".format(roc_auc_score(y_test, predicted_hd)
                                                 ) + "\t\t" + "{:.4f}".format(roc_auc_score(test['result'], test['elo_prob1'])))
print("Brier Score: " + "\t" + "{:.4f}".format(brier_score_loss(y_test, predicted_hd)) +
      "\t\t" + "{:.4f}".format(brier_score_loss(test['result'], test['elo_prob1'])))

# creating a column with the models probabilities to analyze vs elo fivethirtyeight
test.loc[:, 'hm_prob'] = predicted_hd
test = test[['schedule_season', 'schedule_week', 'team_home',
             'team_away', 'elo_prob1', 'hm_prob', 'result']]

# calulate bets won (only make a bet when probability is greater than / equal to 60% or less than / equal to 40%)
test['my_bet_won'] = (((test['hm_prob'] >= 0.60) & (test['result'] == 1)) | (
    (test['hm_prob'] <= 0.40) & (test['result'] == 0))).astype(int)
test['elo_bet_won'] = (((test['elo_prob1'] >= 0.60) & (test['result'] == 1)) | (
    (test['elo_prob1'] <= 0.40) & (test['result'] == 0))).astype(int)

# calulate bets lost (only make a bet when probability is greater than / equal to 60% or less than / equal to 40%)
test['my_bet_lost'] = (((test['hm_prob'] >= 0.60) & (test['result'] == 0)) | (
    (test['hm_prob'] <= 0.40) & (test['result'] == 1))).astype(int)
test['elo_bet_lost'] = (((test['elo_prob1'] >= 0.60) & (test['result'] == 0)) | (
    (test['elo_prob1'] <= 0.40) & (test['result'] == 1))).astype(int)

# printing some quick overall results for my model
print("My Model Win Percentage: " + "{:.4f}".format(test['my_bet_won'].sum() / (
    test['my_bet_lost'].sum() + test['my_bet_won'].sum())))
print("Total Number of Bets Won: " + str(test['my_bet_won'].sum()))
print("Total Number of Bets Made: " +
      str((test['my_bet_lost'].sum() + test['my_bet_won'].sum())))
print("Possible Games: " + str(len(test)))

# printing some quick overall results for fivethirtyeight's ELO model
print("ELO Model Win Percentage: " + "{:.4f}".format(test['elo_bet_won'].sum()/(
    test['elo_bet_lost'].sum() + test['elo_bet_won'].sum())))
print("Total Number of Bets Won: " + str(test['elo_bet_won'].sum()))
print("Total Number of Bets Made: " +
      str((test['elo_bet_lost'].sum() + test['elo_bet_won'].sum())))
print("Possible Games: " + str(len(test)))
