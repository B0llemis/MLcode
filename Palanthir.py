import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator,TransformerMixin

class Palanthir(object):
## Native attributes
    def __init__(self, input, target_feature:str=None, init_test_size:(float,None)=None):
        """Initiates a Palanthir-class on-top a Pandas Dataframe. The class-attributes describes the overall structure and composition of the data"""
        self.input_data = input
        ##When the Palanthir is born with a target variable:
        if isinstance(target_feature,str):
        #...AND is to be split into test-train subsets
            self.Y_col = [target_feature]
            self.X_cols = [col for col in self.input_data.columns if col not in self.Y_col]
            if isinstance(init_test_size,float):
                self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(self.input_data[self.X_cols],self.input_data[self.Y_col],test_size=0.2,random_state=42)
                self.output = self.train_X.copy(deep=True)
        #...BUT IS NOT to be split into test-train subsets
            else:
                self.Y = self.input_data.copy(deep=True)[self.Y_col]
                self.X = self.input_data.copy(deep=True)[self.X_cols]
                self.output = self.X.copy(deep=True)
        ##When the Palanthir is NOT born with a target variable:
        else:
        # ...BUT IS to be split into test-train subsets
            if isinstance(init_test_size,float):
                self.train, self.test = train_test_split(self.input_data,test_size=0.2,random_state=42)
                self.output = self.train.copy(deep=True)
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

    def update_history(self, step=None, snapshot=None,transformer=None,cols=None):
        current_pipeline = self.transformation_history[-1].get('pipeline').get_params().get('transformers')
        ## Pair together columns and transformers from pipelines used in the ColumnTransformer
        trans_col_pairs = [(item,tup[2]) for tup in current_pipeline for item in tup[1].steps]
        ## Insert the new step from added transformer
        step_order = '1' if [step[0] for pip in current_pipeline for step in pip[1].steps] == [] else str(max([int(step[0]) for pip in current_pipeline for step in pip[1].steps]) + 1)
        new_trans = ((step_order,transformer),cols)
        trans_col_pairs += [new_trans]
        ## Explode each transformer and each column for individual trans-on-col-pairs
        col_trans_explode = [(tup[0],col) for tup in trans_col_pairs for col in tup[1]]
        ## Collect to list all transformers on each column
        sort_on_cols = sorted(col_trans_explode,key=lambda l:l[1])
        col_trans_collect = [(key,list(item[0] for item in group)) for key, group in itertools.groupby(sort_on_cols, key=lambda x: x[1])]
        ## Collect to list all columns on each identical list of transformers
        sort_on_steps = sorted(col_trans_collect,key=lambda x: x[1])
        new_trans_col_pairs = [{'columns':list(item[0] for item in group),'transformers':key} for key,group in itertools.groupby(sort_on_steps,key=lambda x: x[1])]
        ## Wrap transformers into pipeline and pipelines into ColumnTransformer
        list_of_params = [(f'CT-{index}',Pipeline(value.get('transformers')),value.get('columns')) for index,value in enumerate(new_trans_col_pairs)]
        ## Update the Transformation History-dictionary
        updatedPipeline = ColumnTransformer(list_of_params)
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
        return self.transformation_history[self.current_version].get('pipeline')

    def declare_target(self,target_feature:str):
        self.current_version += 1
        self.Y_col = [target_feature]
        self.X_cols = [col for col in self.input_data.columns if col not in self.Y_col]
        ## Palanthir has already been split into test-train subsets:
        if hasattr(self,'train') or hasattr(self,'test'):
            self.train_X = self.train[self.X_cols]
            self.train_Y = self.train[self.Y_col]
            self.test_X = self.test[self.X_cols]
            self.test_Y = self.test[self.Y_col]
            self.output = self.train_X.copy(deep=True)
            del self.train, self.test
        ## Palanthir has NOT already been split into test-train subsets:
        else:
            self.Y = self.input_data.copy(deep=True)[self.Y_col]
            self.X = self.input_data.copy(deep=True)[self.X_cols]
            self.output = self.X.copy(deep=True)
        self.update_attributes()
        self.transformation_history.append(
            dict(
                version=self.current_version
                ,transformation=f"Split into X and Y"
                ,result=self.output
                ,pipeline=self.transformation_history[self.current_version-1].get('pipeline'))
        )
        return self.Y_col

    def random_split(self, test_size=0.2):
        """Uses the SKLearn Train_Test_Split to divide the dataset into random training and test subset"""
        from sklearn.model_selection import train_test_split
        self.current_version += 1
        ## Palanthir is already split into X-Y features:
        if hasattr(self,'Y') or hasattr(self,'X'):
            self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(self.X,self.Y,test_size=test_size,random_state=42)
            self.output = self.train_X.copy(deep=True)
        ## Palanthir is NOT already split into X-Y features:
        else:
            self.train, self.test = train_test_split(self.input_data,test_size=test_size,random_state=42)
            self.output = self.train.copy(deep=True)
        self.update_attributes()
        self.transformation_history.append(
            dict(
                version=self.current_version
                ,transformation=f"Split into Test and Train"
                ,result=self.output
                ,pipeline=self.transformation_history[self.current_version-1].get('pipeline'))
        )
        return self.output

## Summarization and description commands
    def summarize(self):
        """Prints the info, description and any missing value-counts for the class"""
        dataset = self.output
        return print(
            "Info: ", dataset.info(),
            "\n",
            "Description: ", dataset.describe(),
            "\n",
            "Missing values: ", dataset.isna().sum()

        )

## Data preprocessing commands

    ## TO BE DEVELOPED
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
        versionCheckpoint = self.current_version if pipeline_version == None else pipeline_version
        pipeline = self.transformation_history[versionCheckpoint].get('pipeline')
        ## Check if a dataset is given at function-runtime
        if isinstance(dataset,pd.core.frame.DataFrame):
            dataset = dataset
            fitted_pipeline = pipeline.fit(self.input_data)
        ## Alternatively check if the Palanthir has already got a X-Y split in its test-train subsets:
        else:
            dataset = self.test_X if hasattr(self,'test_X') else self.test
            fitted_pipeline = pipeline.fit(self.train_X) if hasattr(self,'train_X') else pipeline.fit(self.test)
        self.transformed_test = fitted_pipeline.transform(dataset)
        return self.transformed_test

## Transformation commands
    def PCA(self, n_components=0.80, include_features = [], exclude_features=[],store=True):
        columns = [col for col in self.features_num if col not in exclude_features] if include_features == [] else [col for col in include_features if col not in exclude_features]
        dataset = self.output[columns]
        from sklearn.decomposition import PCA
        transformer = PCA(n_components=n_components).fit(dataset)
        pca_data = transformer.transform(dataset)
        output_df = pd.DataFrame(pca_data, columns=["PCA_" + str(col + 1) for col in range(pca_data.shape[1])],index=dataset.index)
        if store:
            self.output = output_df
            self.update_attributes()
            self.update_history(step="Performed Principal Component Analysis",snapshot=self.output,transformer=transformer,cols=columns)
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
        transformer = SimpleImputer(strategy=strategy).fit(dataset)
        imputed_data = transformer.transform(dataset)
        output_df = pd.DataFrame(imputed_data, columns=dataset.columns, index=dataset.index)
        if store:
            self.output[columns] = output_df
            self.update_attributes()
            self.update_history(step="Filled nulls",snapshot=self.output,transformer=transformer,cols=columns)
        return output_df

    def encode_order(self, include_features = [], exclude_features=[], store=True):
        """Uses the SKLearn OrdinalEncoder to order any categorical features of the dataset"""
        columns = [col for col in self.features_cat if col not in exclude_features] if include_features == [] else [col for col in include_features if col not in exclude_features]
        dataset = self.output[columns]
        from sklearn.preprocessing import OrdinalEncoder
        transformer = OrdinalEncoder().fit(dataset)
        encoded_data = transformer.transform(dataset)
        output_df = pd.DataFrame(encoded_data, columns=dataset.columns, index=dataset.index)
        if store:
            self.output[columns] = output_df
            self.update_attributes()
            self.update_history(step="Encoded order of categorial features",snapshot=self.output,transformer=transformer,cols=columns)
        return output_df

    def make_dummies(self, include_features = [], exclude_features=[], store=True):
        """Uses the SKLearn OneHotEncoder to turn categorical features of the dataset into dummy-variables"""
        columns = [col for col in self.features_cat if col not in exclude_features] if include_features == [] else [col for col in include_features if col not in exclude_features]
        remain_columns = [col for col in self.output.columns if col not in columns]
        dataset = self.output[columns]
        from sklearn.preprocessing import OneHotEncoder
        transformer = OneHotEncoder().fit(dataset)
        new_column_names = transformer.get_feature_names_out(dataset.columns)
        dummy_data = transformer.transform(dataset).toarray()
        dummy_data_df = pd.DataFrame(dummy_data, columns=[name for name in new_column_names], index=dataset.index)
        output_df = pd.merge(self.output[remain_columns], dummy_data_df, left_index=True, right_index=True)
        if store == True:
            self.output = output_df
            self.update_attributes()
            self.update_history(step="Turned categorical features into dummy variables",snapshot=self.output,transformer=transformer,cols=columns)
        return output_df

    def scale(self, strategy:str, include_features = [], exclude_features=[], store=True):
        """Uses the SKLearn StandardScaler or MinMaxScaler to scale all numerical features of the dataset"""
        columns = [col for col in self.features_num if col not in exclude_features] if include_features == [] else [col for col in include_features if col not in exclude_features]
        dataset = self.output[columns]
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        if strategy=="Standard":
            transformer = StandardScaler().fit(dataset)
        elif strategy=="MinMax":
            transformer = MinMaxScaler().fit(dataset)
        else:
            print('Not a proper scaler')
        output_df = transformer.transform(dataset)
        if store:
            self.output[columns] = output_df
            self.update_attributes()
            self.update_history(step=f"""Scaled feature-values using {'Standard-scaler' if strategy=='Standard' else 'MinMax-scaler'}""",snapshot=self.output,transformer=transformer,cols=columns)
        return output_df

    def cluster(self, max_k=10, include_features = [], exclude_features=[], store=True):
        """Uses the SKLearn KMeans to cluster the dataset"""
        from sklearn.base import BaseEstimator,TransformerMixin
        from sklearn.cluster import KMeans
        from sklearn.utils import check_random_state
        from sklearn.metrics import silhouette_score
        from matplotlib import pyplot

        columns = [col for col in self.features_num if col not in exclude_features] if include_features == [] else [col for col in include_features if col not in exclude_features]
        dataset = self.output[columns]
        kmeans_per_k = [KMeans(n_clusters=k, n_init='auto', random_state=42).fit(dataset) for k in range(1, max_k + 1)]
        silhouettes = [silhouette_score(dataset, model.labels_) for model in kmeans_per_k[1:]]
        best_k = silhouettes.index(max(silhouettes)) + 2
        plt.plot(range(2, max_k + 1), silhouettes)
        plt.xlabel("KMeans")
        plt.ylabel("Silhouette-score")
        plt.show()
        print("Best silhouette is obtained with k as: ", best_k)
        if store:
        ## Create Custom Transformer for updating Cluster-label column
            class ClusterIdentifier(BaseEstimator,TransformerMixin):
                def __init__(self, Ks=best_k,random_state=None):
                    self.random_state = random_state
                    self.Ks = Ks

                def fit(self, X, y=None):
                    self.random_state_ = check_random_state(self.random_state)
                    self.estimator = KMeans(n_clusters=self.Ks, n_init='auto', random_state=42).fit(X)
                    return self

                def transform(self, X, y=None):
                    X_trans = pd.DataFrame(data=self.estimator.transform(X)).apply(lambda x: x.astype(float))
                    X_trans['cluster'] = X_trans.idxmin(axis=1)
                    return X_trans

            bestKMeans = KMeans(n_clusters=best_k, random_state=42).fit(dataset)
            #self.output["Cluster"] = ["Cluster " + str(i) for i in bestKMeans.predict(dataset)]
            transformer = ClusterIdentifier().fit(dataset)
            self.output = transformer.transform(X=dataset)
            self.update_attributes()
            self.update_history(step="Added Cluster-label as column to dataset",snapshot=self.output,transformer=transformer,cols=columns)
        return self.output

    def remove_outliers(self, include_features = [], exclude_features = [], factor=1.5):
        from sklearn.base import BaseEstimator,TransformerMixin
        class OutlierRemover(BaseEstimator,TransformerMixin):
            def __init__(self,factor=factor):
                self.factor = factor

            def outlier_detector(self,X,y=None):
                X = pd.Series(X).copy()
                q1 = X.quantile(0.25)
                q3 = X.quantile(0.75)
                iqr = q3 - q1
                self.lower_bound.append(q1 - (self.factor * iqr))
                self.upper_bound.append(q3 + (self.factor * iqr))

            def fit(self,X,y=None):
                self.lower_bound = []
                self.upper_bound = []
                X.apply(self.outlier_detector)
                return self

            def transform(self,X,y=None):
                X = pd.DataFrame(X).copy()
                for i in range(X.shape[1]):
                    x = X.iloc[:, i].copy()
                    x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = np.nan
                    X.iloc[:, i] = x
                return X
        columns = [col for col in self.features_num if col not in exclude_features] if include_features == [] else [col for col in include_features if col not in exclude_features]
        dataset = self.output[columns]



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