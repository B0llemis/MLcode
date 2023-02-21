import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class Palanthir(object):
## Native attributes
    def __init__(self, input, target_feature:str=None, init_test_size:(float,None)=None):
        """Initiates a Palanthir-class on-top a Pandas Dataframe. The class-attributes describes the overall structure and composition of the data"""
        self.input_data = input
        ##When the Palanthir is born with a target variable:
        if isinstance(target_feature,str):
        #...AND is to be split into test-train subsets
            Y = [target_feature]
            X = [col for col in self.input_data.columns if col not in Y]
            if isinstance(init_test_size,float):
                self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(self.input_data[X],self.input_data[Y],test_size=0.2,random_state=42)
                self.output = self.train_X
        #...BUT IS NOT to be split into test-train subsets
            else:
                self.Y = self.input_data[Y]
                self.X = self.input_data[X]
                self.output = self.X
        ##When the Palanthir is NOT born with a target variable:
        else:
        # ...BUT IS to be split into test-train subsets
            if isinstance(init_test_size,float):
                self.train, self.test = train_test_split(self.input_data,test_size=0.2,random_state=42)
                self.output = self.train
        #...AND IS NOT to be split into test-train subsets.
            else:
                self.output = self.input_data.copy(deep=True)
        self.size = len(self.output)
        self.features = list(self.output)
        self.features_num = list(self.output.loc[:, self.output.dtypes != object])
        self.features_cat = list(self.output.loc[:, self.output.dtypes == object])
        self.current_version = 0
        self.transformation_history = [dict(version=0,transformation='input',result=self.input_data,pipeline=ColumnTransformer([]))]

## Self-update and audit commands

    def update_attributes(self):
        self.size = len(self.output)
        self.features = list(self.output)
        self.features_num = list(self.output.loc[:, self.output.dtypes != object])
        self.features_cat = list(self.output.loc[:, self.output.dtypes == object])

    def update_history(self, step=None, snapshot=None,text=None,transformer=None,cols=None):
        pipelineSteps = self.transformation_history[-1].get('pipeline').get_params().get('transformers') + [(text,transformer,cols)]
        updatedPipeline = ColumnTransformer(pipelineSteps)
        self.current_version += 1
        self.transformation_history.append(
            dict(
                version=self.current_version
                ,transformation=step
                ,result=snapshot
                ,pipeline=updatedPipeline
            )
        )

    def restore(self, version=None):
        versionCheckpoint = (self.current_version - 1) if version == None else version
        self.current_version = versionCheckpoint
        self.output = self.transformation_history[versionCheckpoint].get('result')
        self.update_attributes()
        self.transformation_history.append(
            dict(
                version=self.current_version
                ,transformation=f"Restored to version {self.current_version}"
                ,result=self.transformation_history[self.current_version].get('result')
                ,pipeline=self.transformation_history[self.current_version].get('pipeline')
            )
        )

    def declare_target(self,target_feature:str):
        self.Y = self.input_data[[target_feature]]
        self.X = self.input_data[[col for col in self.input_data.columns if col not in self.Y]]
        self.output = self.X
        self.update_attributes()
        self.current_version += 1
        self.transformation_history.append(
            dict(
                version=self.current_version
                ,transformation=f"Split into X and Y"
                ,result=self.output
                ,pipeline=None)
        )
        return self.Y

## Summarization and description commands
    def summarize(self):
        """Prints the info, description and any missing value-counts for the class"""
        dataset = self.output
        return print(
            "Info: ", dataset.info(),
            "Description: ", dataset.describe(),
            "Missing values: ", dataset.isna().sum()
        )

## Data preprocessing commands
    def random_split(self, x=None, y=None, test_size=0.2, store=True):
        """Uses the SKLearn Train_Test_Split to divide the dataset into random training and test subset"""
        dataset_x = x if isinstance(x,pd.core.frame.DataFrame) else self.output.drop(columns=self.Y)
        dataset_y = y if isinstance(y,pd.core.frame.DataFrame) else self.output[self.Y]
        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(dataset_x, dataset_y, test_size=test_size, random_state=42)
        if store:
            self.train_X = train_x
            self.train_Y = train_y
            self.test_X = test_x
            self.test_Y = test_y
        return train_x, train_y, test_x, test_y

    def stratified_split(self, cols, store=True):
        """Uses the SKLearn StratigiesShuffleSplit to divide the dataset into stratified training and test subset"""
        dataset = self.output
        from sklearn.model_selection import StratifiedShuffleSplit
        split = StratifiedShuffleSplit(n_split=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(dataset, dataset[cols]):
            strat_train_set = dataset.loc[train_index]
            strat_test_set = dataset.loc[test_index]
        if store:
            self.train_subset, self.test_subset = [strat_train_set], [strat_test_set]
        return strat_train_set, strat_test_set

    def execute_pipeline(self, dataset=None, pipeline_version=None):
        """Uses the SKLearn ColumnTransformer build via previous transformations and apply its transformations to the target dataset"""
        dataset = self.output if dataset==None else dataset
        versionCheckpoint = self.current_version if pipeline_version == None else pipeline_version
        pipeline = self.transformation_history[versionCheckpoint].get('pipeline')
        self.output = pipeline.transform(dataset)
        return self.output

## Transformation commands
    def PCA(self, n_components=0.80, include_features = [], exclude_features=[],store=True):
        columns = [col for col in self.features_num if col not in exclude_features] if include_features == [] else [col for col in include_features if col not in exclude_features]
        dataset = self.output[columns]
        from sklearn.decomposition import PCA
        PCAtransformer = PCA(n_components=n_components).fit(dataset)
        pca_data = PCAtransformer.transform(dataset)
        output_df = pd.DataFrame(pca_data, columns=["PCA_" + str(col + 1) for col in range(pca_data.shape[1])],index=dataset.index)
        if store:
            self.output = output_df
            self.update_attributes()
            self.update_history(step="Performed Principal Component Analysis",snapshot=self.output,text='pca',transformer=PCAtransformer,cols=columns)
        explained_variance = PCA().fit(dataset).explained_variance_ratio_
        cumsum = np.cumsum(explained_variance)
        print(cumsum)
        plt.plot(["PCA" + str(num) for num in range(1, len(cumsum) + 1)], cumsum)
        plt.show()
        return output_df

    def fill_nulls(self, strategy="median", include_features = [], exclude_features=[], store=True):
        """Uses the SKLearn SimpleImputer to fill out any missing values in the numerical features of the dataset"""
        columns = [col for col in self.features_num if col not in exclude_features] if include_features == [] else [col for col in include_features if col not in exclude_features]
        dataset = self.output[columns]
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy=strategy).fit(dataset)
        imputed_data = imputer.transform(dataset)
        output_df = pd.DataFrame(imputed_data, columns=dataset.columns, index=dataset.index)
        if store:
            self.output[columns] = output_df
            self.update_attributes()
            self.update_history(step="Filled nulls",snapshot=self.output,text='impute',transformer=imputer,cols=columns)
        return output_df

    def encode_order(self, include_features = [], exclude_features=[], store=True):
        """Uses the SKLearn OrdinalEncoder to order any categorical features of the dataset"""
        columns = [col for col in self.features_cat if col not in exclude_features] if include_features == [] else [col for col in include_features if col not in exclude_features]
        dataset = self.output[columns]
        from sklearn.preprocessing import OrdinalEncoder
        encoder = OrdinalEncoder().fit(dataset)
        encoded_data = encoder.transform(dataset)
        output_df = pd.DataFrame(encoded_data, columns=dataset.columns, index=dataset.index)
        if store:
            self.output[columns] = output_df
            self.update_attributes()
            self.update_history(step="Encoded order of categorial features",snapshot=self.output,text='ordinal',transformer=encoder,cols=columns)
        return output_df

    def make_dummies(self, include_features = [], exclude_features=[], store=True):
        """Uses the SKLearn OneHotEncoder to turn categorical features of the dataset into dummy-variables"""
        columns = [col for col in self.features_cat if col not in exclude_features] if include_features == [] else [col for col in include_features if col not in exclude_features]
        remain_columns = [col for col in self.output.columns if col not in columns]
        dataset = self.output[columns]
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder().fit(dataset)
        new_column_names = encoder.get_feature_names_out(dataset.columns)
        dummy_data = encoder.transform(dataset).toarray()
        dummy_data_df = pd.DataFrame(dummy_data, columns=[name for name in new_column_names], index=dataset.index)
        output_df = pd.merge(self.output[remain_columns], dummy_data_df, left_index=True, right_index=True)
        if store == True:
            self.output = output_df
            self.update_attributes()
            self.update_history(step="Turned categorical features into dummy variables",snapshot=self.output,text='onehot',transformer=encoder,cols=columns)
        return output_df

    def scale(self, strategy:str, include_features = [], exclude_features=[], store=True):
        """Uses the SKLearn StandardScaler or MinMaxScaler to scale all numerical features of the dataset"""
        columns = [col for col in self.features_num if col not in exclude_features] if include_features == [] else [col for col in include_features if col not in exclude_features]
        dataset = self.output[columns]
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        if strategy=="Standard":
            scaler = StandardScaler().fit(dataset)
        elif strategy=="MinMax":
            scaler = MinMaxScaler().fit(dataset)
        else:
            print('Not a proper scaler')
        output_df = scaler.transform(dataset)
        if store:
            self.output[columns] = output_df
            self.update_attributes()
            self.update_history(step=f"""Scaled feature-values using {'Standard-scaler' if strategy=='Standard' else 'MinMax-scaler'}""",snapshot=self.output,text='scaler',transformer=scaler,cols=columns)
        return output_df

    def cluster(self, max_k=10, store=True):
        """Uses the SKLearn KMeans to cluster the dataset"""
        dataset = self.output
        from sklearn.cluster import KMeans
        from matplotlib import pyplot
        from sklearn.metrics import silhouette_score
        kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(dataset) for k in range(1, max_k + 1)]
        silhouettes = [silhouette_score(dataset, model.labels_) for model in kmeans_per_k[1:]]
        best_k = silhouettes.index(max(silhouettes)) + 2
        plt.plot(range(2, max_k + 1), silhouettes)
        plt.xlabel("KMeans")
        plt.ylabel("Silhouette-score")
        plt.show()
        print("Best silhouette is obtained with k as: ", best_k)
        if store:
            self.output["Cluster"] = ["Cluster " + str(i) for i in KMeans(n_clusters=best_k, random_state=42).fit_predict(dataset)]
            self.update_attributes()
            self.update_history(step="Added Cluster-label as column to dataset",snapshot=self.output)
        return self.output

## Analysis commands
    def cross_validate(self, model, x, y, score_measure="neg_mean_squared_error", folds=10):
        """Uses the SKLearn Cross_Val_Score to cross-validate one/several models on the training subset"""
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, x, y, scoring=score_measure, cv=folds)
        return scores

    def full_analysis(self, model):
        """Conducts a full data-analysis pipeline on the dataset, including model training, evaluation and tuning"""
        dataset = self.output
        X_train, X_test, Y_train, Y_test = self.random_split(dataset)

        sqrt_scores = np.sqrt(-self.cross_validate(model, X_train, Y_train, score_measure="neg_mean_squared_error", folds=10))
        print(
            "RMSE-scores: ", sqrt_scores,
            "RMSE-mean: ", sqrt_scores.mean(),
            "RMSE-std: ", sqrt_scores.std()
        )