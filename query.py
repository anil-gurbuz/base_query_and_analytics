import logging
import os
import pickle
import sys
import time
import copy

import PIconnect as PI
import pandas as pd
from tqdm import tqdm

from preprocessor import Preprocessor
from preprocessor import QueryArtifacts

try:
    import OSIsoft
except:
    logging.info("Check queryMultipleTimeFrames function if exists -- Might not be able to find OSIsoft.")

tz = "Etc/GMT-10" if time.localtime().tm_isdst == 0 else "Etc/GMT-11"
PI.PIConfig.DEFAULT_TIMEZONE = tz

SERVER = PI.PIServer()

identifier = time.localtime()
identifier = f"{identifier.tm_yday}__{identifier.tm_hour}_{identifier.tm_min}"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# <editor-fold desc="Helper Funcitons">
# Save the object
def savePickle(obj, file_name):
    """
    Save a python object as a .pkl file
    """
    with open(file_name, mode="wb") as file:
        pickle.dump(obj, file)
        file.close()
        logging.info(f"Saved the object to {file_name} successfully!")


# Load the object
def loadPickle(file_name):
    """
    Load a .pkl file
    """
    with open(file_name, mode="rb") as file:
        obj = pickle.load(file)
        file.close()
        return obj


def savePresentableCSV(df, path, recorded=False, remove_g=True, keep_index=True, fillna_with=None):
    if recorded:
        df["tag_name"] = rollbackTagNames(df["tag_name"])

    df.columns = rollbackTagNames(df.columns)
    if remove_g:
        df = df.drop("G", axis=1)

    if fillna_with:
        df = df.fillna(fillna_with)
    df.to_csv(path, index=keep_index)
    logging.info(f"Successfully saved the dataframe to {path} !")


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

# </editor-fold>


# <editor-fold desc="Global Querry Functions">
def resume_another_query(check_point_path):
    q = loadPickle(check_point_path)
    q.run()
    # Save resumed query object regardless  bcs can't return it so the only way is accessing through loading
    savePickle(q, f"{q.save_file_path}_query_obj.pkl")


def getOneInterpolatedTag(tag_name, dates, step_size, periods):
    try:
        tag = SERVER.search(rollbackTagNames(tag_name, True))[0]
    except IndexError as indx_err:
        logging.warning(f"{tag_name} couldn't found. Skipping to other tags for Interpolated Values")
        return None, indx_err.__class__.__name__

    one_tag_series = getMultipleInterpolatedDateRanges(tag, dates, step_size, periods)

    return one_tag_series, tag


def getMultipleInterpolatedDateRanges(tag, dates, step_size, periods):
    multiple_range_series = pd.Series(dtype=str)

    for idx, (start_date, end_date) in enumerate(dates):
        single_range_series = getOneInterpolatedDateRange(tag, start_date, end_date, step_size, periods)
        multiple_range_series = pd.concat([multiple_range_series, single_range_series], axis=0)

    return multiple_range_series


def getOneInterpolatedDateRange(tag, start_date, end_date, step_size, periods):
    """by = ["y", "m", "d", "min", "s"] """

    start_date = pd.to_datetime(start_date, infer_datetime_format=True, dayfirst=True, exact=False)
    end_date = pd.to_datetime(end_date, infer_datetime_format=True, dayfirst=True, exact=False)
    dates = pd.date_range(start_date, end_date, periods=periods, tz=tz)

    single_range_series = pd.Series(dtype=str)

    exception_counter = 0
    i = 0
    while i in range(len(dates) - 1):
        try:
            data = tag.interpolated_values(dates[i], dates[i + 1], step_size)
        except Exception as e:
            if e.__class__.__name__ == "PITimeoutException":
                logging.warning("Timeout exception occurred, re-querying the last date range")
                continue
            elif e.__class__.__name__ == "PIException":
                logging.warning(e)
                sys.exit(
                    "Event collection exceeded the limit error, need to increase freq parameter to avoid. Exception Message above:")
            else:
                logging.warning(
                    f"Exception in getOneInterpolatedDateRange function and name: {e.__class__.__name__} and below is the actual message:")
                logging.info(e)
                exception_counter += 1
                if exception_counter > 10000:
                    sys.exit("Infinite Loop identified! Finalizing the program!")

        single_range_series = pd.concat([single_range_series, pd.Series(data.values.astype(str), index=data.index)],
                                        axis=0)
        i += 1
        logging.info(f"Interpolated Values - Finished period {i}!")

    # Remove duplicate dates
    return single_range_series


def getOneRecordedTag(tag_name, dates, periods):
    """ Difference between interpolated and recorded: Recorded sends tag_name to all the way down to  getOneRange """
    try:
        tag = SERVER.search(rollbackTagNames(tag_name, True))[0]
    except IndexError:
        logging.warning(f"{tag_name} couldn't found. Skipping to other tags for Recorded values")
        return None

    one_tag_df = getMultipleRecordedDateRanges(tag, tag_name, dates, periods)

    return one_tag_df


def getMultipleRecordedDateRanges(tag, tag_name, dates, periods=4):
    multiple_range_df = pd.DataFrame(columns=["g", "DateTime", "tag_name", "record"])
    for (start_date, end_date) in dates:
        single_range_df = getOneRecordedDateRange(tag, tag_name, start_date, end_date, periods=periods)
        multiple_range_df = pd.concat([multiple_range_df, single_range_df], axis=0, ignore_index=True)

    return multiple_range_df


def getOneRecordedDateRange(tag, tag_name, start_date, end_date, periods=4):
    start_date = pd.to_datetime(start_date, infer_datetime_format=True, dayfirst=True, exact=False)
    end_date = pd.to_datetime(end_date, infer_datetime_format=True, dayfirst=True, exact=False)
    dates = pd.date_range(start_date, end_date, periods=periods, tz=tz)

    single_range_df = pd.DataFrame(columns=["g", "DateTime", "tag_name", "record"])

    i = 0
    exception_counter = 0
    while i in range(len(dates) - 1):
        try:
            data = tag.recorded_values(dates[i], dates[i + 1])
        except Exception as e:
            if e.__class__.__name__ == "PITimeoutException":
                logging.warning("Timeout exception occurred, re-querying the last date range")
                continue
            elif e.__class__.__name__ == "PIException":
                logging.info(e)
                sys.exit(
                    "Event collection exceeded the limit error, need to increase freq parameter to avoid. Exception Message above:")
            else:
                logging.warning(
                    f"Exception in getOneInterpolatedDateRange function and name: {e.__class__.__name__} and below is the actual message:")
                logging.info(e)
                exception_counter += 1
                if exception_counter > 10000:
                    sys.exit("Infinite Loop identified! Finalizing the program!")
                continue

        single_range_df = pd.concat([single_range_df, pd.DataFrame(
            {"g": "-1", "DateTime": data.index, "tag_name": tag_name, "record": data.values.astype(str)})], axis=0,
                                    ignore_index=True)
        i += 1
        logging.info(f"Recorded Values - Finished period {i} !")

    return single_range_df


# </editor-fold>
"""


"""
class Query():
    def __init__(self, generate_empty_object=False, **kwargs):
        """
            Example kwargs;
            d = {"tags"               : tags,
                 "dates"              : list(zip([start_date,start_date2], [end_date,end_date2])),
                 "date_names"         : ["before", "after"],
                 "step_size"          : "1m",
                 "periods"            : 6,
                 "save_every_n_tags"  : 10,
                 "interpolated_values": True,
                 "recorded_values"    : False,

                 "global_filter"      : None,
                 "generate_errors"    : False,
                 "generate_cleaned_version"      : False,

                 "save_file_path"     : None,
                 "save_raw_csv"       : False,
                 "save_filtered_csv"  : False,
                 "save_cleaned_csv"   : False,
                 "save_pkl"           : False,

                 "check_point_file_path": None, # To resume the query --> Enough to only supply this parameter.
                 "query_tool_init": False
                 }
        """


        # Raw dfs always initiated -- will be checked if interpolated_values / recorded_values are True or not instead of dfs
        self.raw_int_df = pd.DataFrame(columns=["g", "DateTime"])
        self.raw_rec_df = pd.DataFrame(columns=["g", "DateTime", "tag_name", "record"])

        self.global_flt_int_df = None
        self.cleaned_int_df = None
        self.global_flt_rec_dict = None
        self.cleaned_rec_dict = None


        # Expected via kwargs
        self.tags = []
        self.dates = []
        self.date_names = []
        self.step_size = None
        self.periods = None
        self.save_every_n_tags = None
        self.interpolated_values = False
        self.recorded_values = False
        self.save_file_path = None
        self.save_raw_csv = False,
        self.save_filtered_csv = False
        self.save_cleaned_csv = False
        self.save_pkl = False
        self.check_point_file_path = None
        self.query_tool_init = False

        # Pass to Preprocessor
        self.generate_errors = False
        self.global_filter = None
        self.generate_cleaned_version = False

        # Gets calculated during operations
        self.continuous = []
        self.categorical = []
        self.all_found_tags = []

        self.found_tag_descriptions = {}
        self.found_tag_dtypes = {}

        self.n_tags = 0
        self.interpolated_tag_idx = 0
        self.recoded_tag_idx = 0
        self.n_records = {}

        for (k, v) in kwargs.items():
            self.__setattr__(k, v)

        # Simplify tag names -- Different at query tool
        self.tags = simplifyTagNames(self.tags)

        if generate_empty_object:
            return

        else:
            if self.check_point_file_path:
                resume_another_query(self.check_point_file_path)
            else:
                self.periods = max(self.periods, 2) if self.periods % 2 == 1 else max(self.periods+1, 2)
                self.run()


    def run(self):
        # Get the recorded and interpolated values
        if self.recorded_values:
            self.queryRecordings()

        if self.interpolated_values:
            self.querryInterpolatedTags()

        preprocessor = Preprocessor(self, generate_errors=self.generate_errors, generate_cleaned_version=self.generate_cleaned_version, global_filter=self.global_filter)
        preprocessor.process()
        preprocessed_q_obj = preprocessor.get_query_obj()

        self._generate_artifacts(preprocessed_q_obj)

    def _generate_artifacts(self, preprocessed_q_obj):

        if self.save_file_path:

            if self.save_raw_csv:
                if self.interpolated_values:
                    savePresentableCSV(preprocessed_q_obj.raw_int_df.copy(), f"{self.save_file_path}_raw_int_df.csv",
                                       keep_index=False)
                if self.recorded_values:
                    savePresentableCSV(pd.concat(preprocessed_q_obj.raw_rec_dict).reset_index(drop=True).copy(),
                                       f"{self.save_file_path}_raw_rec_df.csv", keep_index=False, recorded=True)

            if self.save_filtered_csv:
                if self.interpolated_values and self.global_filter:
                    savePresentableCSV(preprocessed_q_obj.global_flt_int_df.copy(),
                                       f"{self.save_file_path}_global_flt_int_df.csv", keep_index=False)
                if self.recorded_values and self.global_filter:
                    savePresentableCSV(pd.concat(preprocessed_q_obj.global_flt_rec_dict).reset_index(drop=True).copy(),
                                       f"{self.save_file_path}_global_flt_rec_df.csv", keep_index=False, recorded=True)

            if self.save_cleaned_csv:
                if self.interpolated_values:
                    savePresentableCSV(preprocessed_q_obj.cleaned_int_df.copy(), f"{self.save_file_path}_cleaned_int_df.csv",
                                       keep_index=False)
                if self.recorded_values:
                    savePresentableCSV(pd.concat(preprocessed_q_obj.cleaned_rec_dict).reset_index(drop=True).copy(),
                                       f"{self.save_file_path}_cleaned_rec_df.csv", keep_index=False, recorded=True)

            if self.save_pkl:
                savePickle(preprocessed_q_obj, f"{self.save_file_path}_query_obj.pkl")

        try:
            os.remove(f"{self.save_file_path}_checkpoint_{identifier}.pkl")
            logging.info("Execution Finished & Checkpoints are removed!")
        except OSError as e:
            logging.info(f"Couldn't find following checkpoint file: {self.save_file_path}_checkpoint_{identifier}.pkl. \nDue to following Suppressed Exception: {e.strerror}")
            logging.info("Query Execution Finished!")

    def querryInterpolatedTags(self):
        self._querryInterpolatedTags(self.tags, self.dates, self.step_size, self.periods)

    def _querryInterpolatedTags(self, tag_names, dates, step_size, periods):
        found_one_tag = False

        pbar = tqdm(total=len(tag_names) - self.interpolated_tag_idx)
        # For all tag_names...
        while self.interpolated_tag_idx < len(tag_names):
            # ...Try querrying all relavant data
            try:
                one_tag_series, tag = getOneInterpolatedTag(tag_names[self.interpolated_tag_idx], dates, step_size, periods)
                one_tag_series = one_tag_series[~one_tag_series.index._duplicated()]

            except AttributeError as attr_err:
                # If tag couldn't found in PI then move to the next on
                if one_tag_series is None and tag == "IndexError":
                    self.interpolated_tag_idx += 1
                    pbar.update(1)
                    continue
            # .... Catch all to see different error cases -- Timeout is handled in getOneInterpolatedDateRange func.
            except Exception as any_exp:
                logging.warning(
                    f"Current exceptions e.__class__.__name__ is {any_exp.__class__.__name__} and below is the actual message:")
                logging.info(any_exp)
                continue

            if (self.interpolated_tag_idx + 1) % self.save_every_n_tags == 0:
                savePickle(self, f"{self.save_file_path}_checkpoint_{identifier}.pkl")

            # If found then add to df and into found tags
            found_one_tag = True
            self._addDataToDF(tag_names[self.interpolated_tag_idx], one_tag_series, tag)
            self.raw_int_df.DateTime = one_tag_series.index
            self.all_found_tags += [tag_names[self.interpolated_tag_idx]]
            self.found_tag_descriptions[tag_names[self.interpolated_tag_idx]] = tag.description
            self.found_tag_dtypes[tag_names[self.interpolated_tag_idx]] = tag.pi_point.PointType
            self.interpolated_tag_idx += 1
            pbar.update(1)

        pbar.close()

        if found_one_tag:
            self._setGroupsOfDateTime(interpolated=True)
            self.n_tags = len(self.all_found_tags)

    def queryRecordings(self):
        self._querryRecordings(self.tags, self.dates, self.periods)

    def _querryRecordings(self, tag_names, dates, periods):

        # For all tags...
        for self.recorded_tag_idx, tag_name in enumerate(tqdm(tag_names)):
            one_tag_df = getOneRecordedTag(tag_name, dates, periods)
            if one_tag_df is None: continue
            self.raw_rec_df = pd.concat([self.raw_rec_df, one_tag_df], axis=0, ignore_index=True)

            if (self.recorded_tag_idx + 1) % self.save_every_n_tags == 0:
                savePickle(self, f"{self.save_file_path}_checkpoint_{identifier}.pkl")

        self._setGroupsOfDateTime(interpolated=False)

        self.n_records = dict(self.raw_rec_df.tag_name.value_counts())

    def _addDataToDF(self, tag_name, one_tag_series, tag):
        if self.raw_int_df.shape[1] != 2:
            assert self.raw_int_df.shape[0] == one_tag_series.shape[0], "Series and DF doesn't have same number of rows"


        if tag.raw_attributes['pointtype'] in ([6,8,12]):
            self.raw_int_df[tag_name] = pd.to_numeric(one_tag_series, errors='coerce')
            self.continuous.append(tag_name)
        else:
            try:
                self.raw_int_df[tag_name] = one_tag_series.astype(str)
            except:
                one_tag_series.name = tag_name
                self.raw_int_df = pd.merge(self.raw_int_df, one_tag_series, how="left", left_index=True, right_index=True)

            self.categorical.append(tag_name)
            logging.warning(f"Datatype for the {tag_name} is found {tag.raw_attributes['pointtype']} hence considered as categorical variable and stored as string!")

        #
        # try:
        #     self.raw_int_df[tag_name] = one_tag_series.astype("float64")
        #     self.continuous.append(tag_name)
        # except ValueError as err:
        #     try:
        #         self.raw_int_df[tag_name] = one_tag_series.astype(str)
        #     except:
        #         one_tag_series.name = tag_name
        #         self.raw_int_df = pd.merge(self.raw_int_df, one_tag_series, how="left", left_index=True, right_index=True)
        #
        #     self.categorical.append(tag_name)
        #     logging.warning(f"While trying to convert {tag_name} to float, BELOW exception is captured and {tag_name} stored as a string")
        #     logging.info(err)

    def _setGroupsOfDateTime(self, interpolated):
        if interpolated:
            if self.raw_int_df.shape[0] == 0:
                logging.info("No interpolated data found!")
            else:
                # --> Depending on pandas version it might have already converted or not to datetime before this step
                if self.raw_int_df.DateTime.dtype == "object":  # Check recroded version for detailed explanation
                    self.raw_int_df["DateTime"] = pd.to_datetime(self.raw_int_df.DateTime)
                self.raw_int_df.g = "-1"
                tz = self.raw_int_df.DateTime.dt.tz
                for idx, (start, end) in enumerate(self.dates):
                    self.raw_int_df.loc[((self.raw_int_df.DateTime >= pd.to_datetime(start, infer_datetime_format=True,
                                                                                     dayfirst=True,
                                                                                     exact=False).tz_localize(tz)) & (
                                                 self.raw_int_df.DateTime <= pd.to_datetime(end,
                                                                                            infer_datetime_format=True,
                                                                                            dayfirst=True,
                                                                                            exact=False).tz_localize(
                                             tz))), "g"] = self.date_names[idx]
        else:
            if self.raw_rec_df.shape[0] == 0:
                logging.info("No recorded data found!")
            else:
                if self.raw_rec_df.DateTime.dtype == "object":
                    # recording_df directly create by appending tag.recorded_values which has a fixed datetime&tz style which is compatible with pd.to_datetime
                    # Hence, we can directly apply it on the recording_df to make sure datetime column is not object--> recording_df initiated seperately and has defulat object columns
                    # --> Depending on pandas version it might have already converted or not to datetime before this step
                    self.raw_rec_df["DateTime"] = pd.to_datetime(self.raw_rec_df.DateTime)

                tz = self.raw_rec_df.DateTime.dt.tz
                for idx, (start, end) in enumerate(self.dates):
                    self.raw_rec_df.loc[
                        ((self.raw_rec_df.DateTime >= pd.to_datetime(start, infer_datetime_format=True, dayfirst=True,
                                                                     exact=False).tz_localize(tz)) & (
                                 self.raw_rec_df.DateTime <= pd.to_datetime(end, infer_datetime_format=True,
                                                                            dayfirst=True, exact=False).tz_localize(
                             tz))), "g"] = \
                        self.date_names[idx]

    def addNewTags(self, new_tags):
        if self.interpolated_values:
            assert type(self.raw_int_df.index[0]) == pd.Timestamp, "To add new tags, indexes should be Timestamp to find matching entries of this new tags"
            if self.periods == 1:
                self.periods = 2

            self._querryInterpolatedTags(new_tags, self.dates, self.step_size, self.periods)

        if self.recorded_values:
            self._querryRecordings(new_tags, self.dates, self.periods)

        preprocessor = Preprocessor(self, generate_errors=self.generate_errors,
                                    generate_cleaned_version=self.generate_cleaned_version,
                                    global_filter=self.global_filter)
        preprocessor.process()
        preprocessed_q_obj = preprocessor.get_query_obj()

        return preprocessed_q_obj

    # TODO: Convert Querry method to PreProcessor OR find a way to produce processed format of the data
    def addNewDates(self, dates, date_names, step_size, periods):
        d = {"controllers"    : [cont.split("_")[0] for cont in self.controllers],
             "other_tags"     : self.tags,
             "dates"          : dates,
             "date_names"     : date_names,
             "step_size"      : step_size,
             "periods"        : periods,
             "recorded_values": self.recorded_values,
             }

        q_tmp = Query(**d)
        new_dates_cols = q_tmp.df.columns.to_list()
        existing_cols = self.raw_int_df.columns.to_list()

        if step_size != self.step_size:
            logging.info(
                f"INFO: Initial step size was {self.step_size} and step size for newly added dates is {step_size}")

        if not all([col in new_dates_cols for col in existing_cols]):
            logging.info(
                "Columns are not matching for existing and new dates. Returning Querry object without attaching the queried df to existing one")
            return q_tmp

        self.raw_int_df = self.raw_int_df.append(q_tmp.df, ignore_index=True)
        logging.info("Appended newly queried dates! Returned Querry object.")
        return q_tmp

    def removeTags(self, tags_to_remove):
        # Remove each tag from list of tags
        for tag in tags_to_remove:
            self.continuous = [cont for cont in self.continuous if cont != tag]
            self.categorical = [cat for cat in self.categorical if cat != tag]

            self.all_found_tags = self.continuous + self.categorical

            if self.n_tags == len(self.all_found_tags):
                logging.info(f"{tag} couldn't found in the tag list")

        # Remove tags from the dataframes
        self.raw_int_df = self.raw_int_df[["g", "DateTime"] + self.all_found_tags]
        self.raw_rec_df = self.raw_rec_df.loc[self.raw_rec_df.tag_name.isin(self.all_found_tags),]

    def getNRecordsInTimeRange(self, start_date, end_date):
        """If using string input for dates. Can use following Format 2020-10-20 14:00:00"""
        filtered_df = self.getDateRange(start_date, end_date, interpolated=False, inplace=False)
        n_records = dict(filtered_df.tag_name.value_counts)
        logging.info("N Records:", n_records)
        return n_records

    def getTags(self, tags: list, interpolated=True):
        # Can only get tags that are found
        tags = [tag for tag in tags if tag in self.all_found_tags]
        if interpolated:
            return self.raw_int_df[["g", "DateTime"] + tags]
        else:
            return self.raw_rec_df.loc[self.raw_rec_df.tag_name.isin(tags),]

    def getOtherTags(self, interpolated=True):
        return self.getTags(self.tags, interpolated)

    def getContinuous(self, interpolated=True):
        return self.getTags(self.continuous, interpolated)

    def getCategorical(self, interpolated=True):
        return self.getTags(self.categorical, interpolated)

    def getDateRange(self, start_date, end_date, interpolated=True, inplace=False):
        """If using string input for dates. Can use following Format 2020-10-20 14:00:00"""
        tz = self.raw_int_df["DateTime"].dt.tz

        if type(start_date) == str:
            start_date = pd.Timestamp(start_date, tz=tz)
        if type(end_date) == str:
            end_date = pd.Timestamp(end_date, tz=tz)

        if interpolated:
            if inplace:
                self.raw_int_df = self.raw_int_df.loc[
                    (self.raw_int_df["DateTime"] > start_date) & (self.raw_int_df["DateTime"] < end_date),]
                return self.raw_int_df
            else:
                return self.raw_int_df.loc[
                    (self.raw_int_df["DateTime"] > start_date) & (self.raw_int_df["DateTime"] < end_date),]
        else:
            if inplace:
                self.raw_rec_df = self.raw_rec_df.loc[
                    (self.raw_rec_df["DateTime"] > start_date) & (self.raw_rec_df["DateTime"] < end_date),]
                self.n_records = self.raw_rec_df.shape[0]
            else:
                return self.raw_rec_df.loc[
                    (self.raw_rec_df["DateTime"] > start_date) & (self.raw_rec_df["DateTime"] < end_date),]

    def attach_query_obj(self, q2):

        assert (self.raw_int_df.index != q2.raw_int_df.index).sum() == 0, "Can not combine query objects without matching indexes!"

        # Combine raw_int_df
        self.raw_int_df = pd.concat([self.raw_int_df, q2.raw_int_df], axis=1)
        # Remove the duplicated columns
        self.raw_int_df = self.raw_int_df.iloc[:, ~self.raw_int_df.columns.duplicated()]

        # Combine Raw recorded
        if q2.raw_rec_df is not None and self.raw_rec_df is not None:
            self.raw_rec_df = pd.concat([self.raw_rec_df, q2.raw_rec_df], axis=0).sort_values('DateTime').reset_index(
                drop=True)

        if q2.raw_rec_df is not None and self.raw_rec_df is None:
            self.raw_rec_df = q2.raw_rec_df

        # Combine global filters and tags etc.
        self.tags +=  q2.tags
        self.global_filter = f'{self.global_filter} , {q2.global_filter}'

        self.all_found_tags = self.artifacts.queried + q2.artifacts.queried
        self.found_tag_descriptions = {**self.artifacts.description_mapper, **q2.artifacts.description_mapper}
        self.found_tag_dtypes = {**self.found_tag_dtypes, **q2.found_tag_dtypes}


        preprocessor = Preprocessor(self, generate_errors=self.generate_errors,
                                    generate_cleaned_version=self.generate_cleaned_version,
                                    global_filter=self.global_filter)
        preprocessor.process()
        preprocessed_q_obj = preprocessor.get_query_obj()

        self._generate_artifacts(preprocessed_q_obj)

    def get_part_of_query_obj(self, tags: list):
        """Returns a new query object with only the tags specified in the list"""
        copied_q_obj = copy.deepcopy(self)

        tags_to_select = list(set(["g", "DateTime"] + tags))
        copied_q_obj.tags = [tag for tag in tags_to_select if tag not in ["g", "DateTime"]]
        copied_q_obj.all_found_tags = [tag for tag in tags_to_select if tag not in ["g", "DateTime"]]
        # Modify artifacts for the new q_obj
        copied_q_obj.found_tag_descriptions = {k: v for (k, v) in copied_q_obj.artifacts.description_mapper.items() if k in tags_to_select}
        copied_q_obj.found_tag_dtypes = {k: v for (k, v) in copied_q_obj.found_tag_dtypes.items() if k in tags_to_select}
        copied_q_obj.artifacts.cont = [tag for tag in copied_q_obj.artifacts.cont if tag in tags_to_select]
        copied_q_obj.artifacts.categorical = [tag for tag in copied_q_obj.artifacts.categorical if tag in tags_to_select]
        copied_q_obj.artifacts.all_tags = [tag for tag in self.artifacts.all_tags if tag in tags_to_select]

        copied_q_obj.artifacts.pv=[tag for tag in self.artifacts.pv if tag in tags_to_select]
        copied_q_obj.artifacts.mv=[tag for tag in self.artifacts.mv if tag in tags_to_select]
        copied_q_obj.artifacts.sv=[tag for tag in self.artifacts.sv if tag in tags_to_select]
        copied_q_obj.artifacts.mode=[tag for tag in self.artifacts.mode if tag in tags_to_select]
        copied_q_obj.artifacts.dv=[tag for tag in self.artifacts.dv if tag in tags_to_select]
        copied_q_obj.artifacts.dv_abs=[tag for tag in self.artifacts.dv_abs if tag in tags_to_select]
        copied_q_obj.artifacts.base_to_tags_mapper = {k:[] for k in self.artifacts.base_to_tags_mapper.keys()}

        for k,v in self.artifacts.base_to_tags_mapper.items():
            for tag in v:
                if tag in tags_to_select:
                    copied_q_obj.artifacts.base_to_tags_mapper[k].append(tag)
        copied_q_obj.artifacts.base_to_tags_mapper = {k:v for k,v in copied_q_obj.artifacts.base_to_tags_mapper.items() if len(v)>0}

        # Modify relevant interpolated dataframes for the new q_obj
        if  copied_q_obj.raw_int_df  is not None:
            copied_q_obj.raw_int_df = copied_q_obj.raw_int_df[tags_to_select]
        if  copied_q_obj.global_flt_int_df is not None:
            copied_q_obj.global_flt_int_df = copied_q_obj.global_flt_int_df[tags_to_select]
        if  copied_q_obj.cleaned_int_df is not None:
            copied_q_obj.cleaned_int_df = copied_q_obj.cleaned_int_df[tags_to_select]

        # Modify relevant recorded dataframe/dict for the new q_obj
        if copied_q_obj.raw_rec_df is not None:
            copied_q_obj.raw_rec_df = copied_q_obj.raw_rec_df.loc[copied_q_obj.raw_rec_df.tag_name.isin(tags_to_select),].reset_index(drop=True)
        if copied_q_obj.global_flt_rec_dict is not None:
            copied_q_obj.raw_rec_df = {k:v for (k,v) in copied_q_obj.global_flt_rec_dict.items() if k in tags_to_select}
        if copied_q_obj.cleaned_rec_dict is not None:
            copied_q_obj.cleaned_rec_dict = {k:v for (k,v) in copied_q_obj.cleaned_rec_dict.items() if k in tags_to_select}


        return copied_q_obj

    def sortArtifacts(self):
        """
        - Needs to be the last step
        - Requires artifacts to be generated -- almost meaning that preprocessing had run before
        - Mirror of Preprocessor._sortColumnsAndMetaInfo() for
        """
        # if no artifacts generated then doesn't run the function
        if not hasattr(self, 'artifacts'):
            return None

        # Sort columns
        if self.interpolated_values:
            ordered_columns = ["g", "DateTime"] + self.raw_int_df.columns[2:].sort_values().to_list()
            self.raw_int_df = self.raw_int_df[ordered_columns]
            self.cleaned_int_df = self.cleaned_int_df[ordered_columns] if self.cleaned_int_df is not None else None
            if self.global_flt_int_df is not None:
                self.global_flt_int_df = self.global_flt_int_df[ordered_columns]

        self.artifacts.cont.sort()
        self.artifacts.categorical = ["g"] + sorted([col for col in self.categorical if col != "g"])

        self.artifacts.all_tags.sort()
        self.artifacts.all_found_tags.sort()
        self.artifacts.all_generated_tags.sort()

        self.artifacts.pv.sort()
        self.artifacts.mv.sort()
        self.artifacts.sv.sort()
        self.artifacts.mode.sort()
        self.artifacts.others.sort()
        self.artifacts.dv.sort()
        self.artifacts.dv_abs.sort()

        self.artifacts.base_to_tags_mapper = dict(sorted(self.artifacts.base_to_tags_mapper.items()))


    def derive_new_tag(self, tag_name, tag_decription, raw_derived_values, global_flt_derived_values=None, categorical=False):
        '''
        - ONLY SUPPORTS INTERPOLATED VALUES
        - ONLY SUPPORTS IGNORES CLEANED INT DF
        categorical: Set true if the derived variable is categorical
        :return:
        :rtype:
        '''

        tag_name = simplifyTagNames(tag_name,accept_single_name=True)
        self.raw_int_df[tag_name] = raw_derived_values

        if self.global_flt_int_df is not None:
            assert global_flt_derived_values is not None, 'There is a global filter applied to query object. Hence, need to supply derived tag"s global flt version as well!'

            self.global_flt_int_df[tag_name] = global_flt_derived_values


        # Modify artifacts
        self.artifacts.rolledback_tag_mapper[tag_name] = rollbackTagNames(tag_name, True)
        self.artifacts.description_mapper[tag_name] = tag_decription
        if categorical:
            self.artifacts.categorical += [tag_name]
        else:
            self.artifacts.cont += [tag_name]


        self.sortArtifacts()
    @staticmethod
    def generate_q_obj_from_data_frame(df, categorical_col_names=[], tag_name_description_mapper=None):
        '''
        - Requires df to have a column named DateTime and g
        - Works only for interpolated values
        - Assumes all the tags are pv and appends a _pv to the column names
        - Tag descriptions are assumed to be the column names
        '''


        q = Query(generate_empty_object=True)

        if 'g' not in categorical_col_names:
            categorical_col_names.append('g')

        q.raw_int_df = df
        q.all_tags = [col for col in df.columns if col not in ['DateTime','g']]
        q.same_base = {}
        q.mv = [tag  for tag in q.all_tags if tag.endswith('mv')]
        q.sv = [tag  for tag in q.all_tags if tag.endswith('sv')]
        q.mode = [tag  for tag in q.all_tags if tag.endswith('mode')]
        q.dv = [tag  for tag in q.all_tags if tag.endswith('dv')]
        q.dv_abs = [tag  for tag in q.all_tags if tag.endswith('dv_abs')]
        q.pv = [tag for tag in q.all_tags if tag not in q.mv + q.sv + q.mode + q.dv + q.dv_abs]

        q.categorical = categorical_col_names
        q.continuous = [col for col in q.all_tags if col not in categorical_col_names]
        q.all_generated_tags = []
        q.all_found_tags = list(q.all_tags)
        if tag_name_description_mapper is not None:
            q.found_tag_descriptions = tag_name_description_mapper
        else:
            q.found_tag_descriptions = dict(zip(q.all_found_tags, q.all_found_tags))
        q.date_names = list(q.raw_int_df.g.unique())
        q.artifacts = QueryArtifacts(q)

        return q

