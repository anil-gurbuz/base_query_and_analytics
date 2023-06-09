import pickle

import numpy as np
import pandas as pd


# <editor-fold desc="Helper Functions">
# Save the object
def savePickle(obj, file_name):
    """
    Save a python object as a .pkl file
    """
    with open(file_name, mode="wb") as file:
        pickle.dump(obj, file)
        file.close()
        print(f"Saved the object to {file_name} successfully!")


# Load the object
def loadPickle(file_name):
    """
    Load a .pkl file
    """
    with open(file_name, mode="rb") as file:
        obj = pickle.load(file)
        file.close()
        return obj


def simplifyTagNames(tag_names, accept_single_name=False):
    if accept_single_name:
        return tag_names.lower().replace(".", "_")
    return [tag_name.lower().replace(".", "_") for tag_name in tag_names]


def rollbackTagNames(tag_names, accept_single_name=False):
    if accept_single_name:
        if tag_names.endswith("_qa"):
            return tag_names[:-3].upper().replace("_", ".") + "_QA"
        else:
            return tag_names.upper().replace("_", ".")

    return [ tag_name[:-3].upper().replace("_", ".") + "_QA" if tag_name.endswith("_qa") else tag_name.upper().replace("_", ".") for tag_name in tag_names]


def getOutlierFilter(df, col_name):
    q1 = df[col_name].describe()["25%"]
    q3 = df[col_name].describe()["75%"]
    iqr = q3 - q1
    flt = (df[col_name] <= (q3 + 1.5 * iqr)) & (df[col_name] >= (q1 - 1.5 * iqr))
    return flt


# </editor-fold>

# <editor-fold desc="Extra Functions">
# generate day stats
def get_date_start(df, date_column="DateTime"):
    day_start = pd.Series(index=df.index, data=np.NAN)
    day_start.loc[(df[date_column].dt.hour >= 7) & (df[date_column].dt.hour < 19)] = \
        df.loc[(df[date_column].dt.hour >= 7) & (df[date_column].dt.hour < 24)][date_column].dt.strftime(
            "%Y-%m-%d 07:00:00")
    day_start.loc[(df[date_column].dt.hour >= 0) & (df[date_column].dt.hour < 7)] = (
            df.loc[(df[date_column].dt.hour >= 0) & (df[date_column].dt.hour < 7)][date_column] - pd.to_timedelta(1,
                                                                                                                  unit="day")).dt.strftime(
        "%Y-%m-%d 07:00:00")
    return day_start


def get_shift_start(df, date_column="DateTime"):
    shift_start = pd.Series(index=df.index, data=np.NAN)
    shift_start.loc[(df[date_column].dt.hour >= 7) & (df[date_column].dt.hour < 19)] = \
        df.loc[(df[date_column].dt.hour >= 7) & (df[date_column].dt.hour < 19)][date_column].dt.strftime(
            "%Y-%m-%d 07:00:00")
    shift_start.loc[(df[date_column].dt.hour >= 19) & (df[date_column].dt.hour < 24)] = \
        df.loc[(df[date_column].dt.hour >= 19) & (df[date_column].dt.hour < 24)][date_column].dt.strftime(
            "%Y-%m-%d 19:00:00")
    shift_start.loc[(df[date_column].dt.hour >= 0) & (df[date_column].dt.hour < 7)] = (
            df.loc[(df[date_column].dt.hour >= 0) & (df[date_column].dt.hour < 7)][date_column] - pd.to_timedelta(1, unit="day")).dt.strftime("%Y-%m-%d 19:00:00")

    return shift_start
# </editor-fold>


class Preprocessor():
    def __init__(self, query_obj, generate_errors=True, generate_cleaned_version=True, global_filter=None):
        """
        Assumes simplified tag names

        "global_filter"
            - If callable --> Should take query object as argument and apply filter on query object
                                --  (must have 2 parts for recorded and interpolated and needs to accept recorded=True/False) int/rec if q_obj have and return filtered q_obj
            - If string --> converted into list of lists and will be applied for int/rec seperately
        """
        self.query = query_obj
        self.generate_errors = generate_errors
        self.generate_cleaned_version = generate_cleaned_version

        if global_filter is not None and global_filter !='':
            self.global_filter = [[simplifyTagNames(flt.split()[0],True).strip(), flt.split()[1].strip(), flt.split()[2].strip()] for flt in global_filter.split(",")] if type(global_filter) == str else global_filter
        else:
            self.global_filter = None




        ####### Interpolated ######
        # Raw (raw_int_df)
        # Custom Filtered (global_flt_int_df)
        # Cleaned: Custom Filtered & outliers replaced with NA(clean_int_df)


        ###### Recorded ######
        # Raw (raw_rec_df/dict)
        # Custom Filtered (global_flt_rec_dict)
        # Cleaned: Custom Filtered & outliers replaced with NA(clean_rec_dict)


    def process(self):
        self._generateGroupsOfTags()

        if self.query.interpolated_values:
            self._process_interpolated()

        if self.query.recorded_values:
            self._process_recorded()

        ## General -- Generate & Organize artifacts
        self._sortColumnsAndMetaInfo()
        # Generate artifacts
        self.query.artifacts = QueryArtifacts(self.query)
        # Clean-up redundant artifacts
        self._cleanup_parent_artifacts()

    def _apply_global_filter(self, recorded=False):
        """
        # Accepts following operators: "<", ">", "=", "<=", ">=", "!="
        # OR a callable as a custom filter --> Callable must accept a query obj and return a filtered query object
        """
        operators_dict = {"<" : (lambda x, y: x < y), ">": (lambda x, y: x > y), "=": (lambda x, y: x == y),
                          "<=": (lambda x, y: x <= y), ">=": (lambda x, y: x >= y), "!=": (lambda x, y: x != y)}
        for tag, operator, value in self.global_filter:
            if recorded:
                self.query.global_flt_rec_dict = {
                    k: (v.loc[operators_dict[operator](v.record, pd.to_numeric(value,errors="ignore")),] if k == tag else v) for (k, v) in
                    self.query.global_flt_rec_dict.items()}
            else:
                self.query.global_flt_int_df = self.query.global_flt_int_df.loc[operators_dict[operator](self.query.global_flt_int_df[tag], pd.to_numeric(value,errors="ignore")),]

    def get_query_obj(self):
        return self.query

    def _generateGroupsOfTags(self):
        """
        Needs to be the first called function
        """

        # Queried / Generated & all tags
        # self.query.all_found_tags
        self.query.all_generated_tags = []
        self.query.all_tags = self.query.all_found_tags + self.query.all_generated_tags

        # Categorical / Continuous
        self.query.continuous = [k for (k, v) in self.query.found_tag_dtypes.items() if
                                 v in [6, 8, 11, 12, 13]]  # 6 & 8 integer, 11-12-13 float
        self.query.categorical = [k for (k, v) in self.query.found_tag_dtypes.items() if v in [101, 105]] + ["g"]

        # Logical Groups
        self.query.pv = [tag for tag in self.query.all_found_tags if tag.endswith("_pv")]
        self.query.mv = [tag for tag in self.query.all_found_tags if tag.endswith("_mv")]
        self.query.sv = [tag for tag in self.query.all_found_tags if tag.endswith("_sv")]
        self.query.mode = [tag for tag in self.query.all_found_tags if tag.endswith("_mode")]
        self.query.others = [tag for tag in self.query.all_found_tags if
                             tag not in self.query.pv + self.query.mv + self.query.sv + self.query.mode]

        self.query.dv = []
        self.query.dv_abs = []

        # Same base dictionary
        # Split tags from "_" and remove the last bit and join the first parts back then make those keys of a dict.
        self.query.same_base = {k: [] for k in
                                list(set(["_".join(tag.split("_")[:-1]) for tag in self.query.all_found_tags]))}
        # Select the same base tags from all_tags
        self.query.same_base = {k: v + [tag for tag in self.query.all_tags if tag.startswith(k)] for (k, v) in
                                self.query.same_base.items()}

    def _cleanup_parent_artifacts(self):
        del self.query.found_tag_descriptions, self.query.all_generated_tags, self.query.all_found_tags, self.query.all_tags, self.query.same_base, self.query.pv, self.query.mv, self.query.sv, self.query.mode, self.query.dv, self.query.dv_abs, self.query.categorical, self.query.continuous,

    # <editor-fold desc="Interpolated df Functions">

    def _process_interpolated(self):
        self._adjustInterpolatedDFDataTypes()

        # Generate error terms for raw_int_df
        if self.generate_errors:
            self._generateErr()

        # Apply custom filter
        if self.global_filter:
            self.query.global_flt_int_df = self.query.raw_int_df.copy()
            if type(self.global_filter) == list:
                self._apply_global_filter(recorded=False)
            elif callable(self.global_filter):
                self.query = self.global_filter(self.query, recorded=False)
        else:
            self.query.global_flt_int_df = None

        if self.generate_cleaned_version:
            self._generateCleanedInterpolated()

    def _adjustInterpolatedDFDataTypes(self):
        self.query.raw_int_df[self.query.continuous] = self.query.raw_int_df[self.query.continuous].apply(
            lambda x: pd.to_numeric(x, "coerce"))

    def _generateErr(self):

        for base, tags in self.query.same_base.items():
            # If any tag of the same base is in both pv & sv
            if any([tag in self.query.pv for tag in tags]) & any([tag in self.query.sv for tag in tags]):
                self.query.raw_int_df[base + "_dv"] = self.query.raw_int_df[base + "_pv"] - self.query.raw_int_df[
                    base + "_sv"]
                self.query.raw_int_df[base + "_dv_abs"] = abs(self.query.raw_int_df[base + "_dv"])

                self.query.dv += [base + "_dv"]
                self.query.dv_abs += [base + "_dv_abs"]
                self.query.same_base[base] += [base + "_dv", base + "_dv_abs"]
                self.query.continuous += [base + "_dv", base + "_dv_abs"]
                self.query.all_generated_tags += [base + "_dv", base + "_dv_abs"]
                self.query.all_tags += [base + "_dv", base + "_dv_abs"]

    def _generateCleanedInterpolated(self):
        self.query.cleaned_int_df = self.query.global_flt_int_df.copy() if self.query.global_flt_int_df is not None else self.query.raw_int_df.copy()
        for col in self.query.continuous:
            flt = getOutlierFilter(self.query.raw_int_df, col)
            self.query.cleaned_int_df.loc[~flt, col] = np.NAN

    def _sortColumnsAndMetaInfo(self):
        """
        Needs to be the last step
        """
        # Sort columns
        if self.query.interpolated_values:
            ordered_columns = ["g", "DateTime"] + self.query.raw_int_df.columns[2:].sort_values().to_list()
            self.query.raw_int_df = self.query.raw_int_df[ordered_columns]
            self.query.cleaned_int_df = self.query.cleaned_int_df[ordered_columns] if self.query.cleaned_int_df is not None else None
            if self.query.global_flt_int_df is not None:
                self.query.global_flt_int_df = self.query.global_flt_int_df[ordered_columns]

        self.query.continuous.sort()
        self.query.categorical = ["g"] + sorted([col for col in self.query.categorical if col != "g"])

        self.query.all_tags.sort()
        self.query.all_found_tags.sort()
        self.query.all_generated_tags.sort()

        self.query.pv.sort()
        self.query.mv.sort()
        self.query.sv.sort()
        self.query.mode.sort()
        self.query.others.sort()
        self.query.dv.sort()
        self.query.dv_abs.sort()

        self.query.same_base = dict(sorted(self.query.same_base.items()))

    def getFreqAgg(self, freq="5min", grouper=None):
        if grouper is None:
            grouper = self.query.raw_int_df.DateTime.dt.floor(freq=freq)

        tmp = self.query.raw_int_df.groupby(grouper)[self.query.continuous].mean()
        tmp2 = self.query.raw_int_df.groupby(grouper)[self.query.categorical].agg(lambda x: x.value_counts().index[0])
        tmp = pd.concat([tmp, tmp2], axis=1)
        tmp = tmp[["g"] + sorted([col for col in self.query.categorical + self.query.continuous if col != "g"])]

        return tmp

    # </editor-fold>

    # <editor-fold desc="Recorded df Transformations">

    def _process_recorded(self):
        self._generateSecondsBtwRecordings()
        self._recordingDfToDict()
        self._adjustRecordedDFDataTypes()

        # Apply custom filter
        if self.global_filter:
            self.query.global_flt_rec_dict = self.query.raw_rec_dict.copy()
            if type(self.global_filter) == list:
                self._apply_global_filter(recorded=True)
            elif callable(self.global_filter):
                self.query = self.global_filter(self.query, recorded=True)
        else:
            self.query.global_flt_rec_dict = None

        if self.generate_cleaned_version:
            self._generateCleanedRecorded()

    def _generateSecondsBtwRecordings(self):
        self.query.raw_rec_df["sec_btw"] = self.query.raw_rec_df.groupby(
            ["g", "tag_name"]).DateTime.diff().dt.seconds.shift(-1).values

    def _recordingDfToDict(self):
        self.query.raw_rec_dict = {
            k: self.query.raw_rec_df.loc[self.query.raw_rec_df.tag_name == k].reset_index(drop=True) for k in
            self.query.raw_rec_df.tag_name.unique()}
        self.raw_rec_df = None

    def _adjustRecordedDFDataTypes(self):
        """
            Works after converting to dictionary only
            """
        for tag, df in self.query.raw_rec_dict.items():
            if tag in self.query.continuous:
                df["record"] = pd.to_numeric(df["record"], errors="coerce")
                self.query.raw_rec_dict[tag] = df

    def _generateCleanedRecorded(self):
        """
        Works after converting to dictionary only
        """
        if self.query.global_flt_rec_dict:
            self.query.cleaned_rec_dict = {
                k: (v.loc[getOutlierFilter(v, "record"),] if k in self.query.continuous else v) for (k, v) in
                self.query.global_flt_rec_dict.items()}
        else:
            self.query.cleaned_rec_dict = {
                k: (v.loc[getOutlierFilter(v, "record"),] if k in self.query.continuous else v) for (k, v) in
                self.query.raw_rec_dict.items()}

    def getTimeWeightedAvg(self):
        """
        Works after converting to dictionary only
        """
        pass

    # </editor-fold>


class QueryArtifacts:
    # simplified tag - description mapper
    # simplified tag - original tag mapper
    # simplified base - list of simplified tags mapper
    # list of simplified tags - simplilfied base mapper -- Opposite of the previous
    # list of base, pv, mv, sv, mode, dv, dv_abs,
    # list of continuous & categorical
    # list of queried & generated - all tags

    def __init__(self, query_obj, important_tags=[]):
        self.important_tags = important_tags

        # tag to rolledback tag
        self.rolledback_tag_mapper = dict(zip(query_obj.all_tags, rollbackTagNames(query_obj.all_tags)))
        # base name to relevant tags list mapper
        self.base_to_tags_mapper = query_obj.same_base
        # tag name to base name mapper
        self.tags_to_base_mapper = dict(zip(sum(list(query_obj.same_base.values()), []),
                                            sum([[base] * len(tags) for base, tags in query_obj.same_base.items()],
                                                [])))
        # Lists of variables
        self.pv = query_obj.pv
        self.mv = query_obj.mv
        self.sv = query_obj.sv
        self.mode = query_obj.mode
        self.dv = query_obj.dv
        self.dv_abs = query_obj.dv_abs

        self.categorical = query_obj.categorical
        self.cont = query_obj.continuous

        self.generated = query_obj.all_generated_tags
        self.queried = query_obj.all_found_tags
        self.all_tags = query_obj.all_tags

        # tag to description
        dv_desc={corresponding_pv[:-2] + 'dv': 'Error Calc - ' + query_obj.found_tag_descriptions[corresponding_pv] for corresponding_pv in [tag[:-2] + 'pv' for tag in self.dv]}
        dv_abs_desc={corresponding_pv[:-2] + 'dv_abs': 'Absolute Error Calc - ' + query_obj.found_tag_descriptions[corresponding_pv] for corresponding_pv in [tag[:-6] + 'pv' for tag in self.dv_abs]}
        self.description_mapper = {**query_obj.found_tag_descriptions,**dv_desc,**dv_abs_desc}
