import logging
import itertools
import numpy as np
import pandas as pd
import warnings
import yaml
from base_query_and_analytics.query import savePresentableCSV, simplifyTagNames
from jinja2 import FileSystemLoader, Environment
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=UserWarning)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.stats import ttest_ind
from scipy.stats import t
import statistics, math
from base_query_and_analytics.global_utils import generate_shifts, generate_shift_day

# TODO: Add local_flt_int_df --> Filters will only generate NAs for interpolated values to av

def load_yml(file_name):
    with open(file_name, "r") as file:
        return yaml.full_load(file)


class Analytics():
    """
    - Only works for interpolated data atm.
    - Global filter is applied to all columns together so removes the rows
    - Local filter is applied each column seperately so replaces entries with NAs

    ## Query object dataframes
    # raw_int_df
    # global_flt_int_df
    # cleaned_int_df

    #### Analytics object
    # local_flt_int_df
    # agg_df_{agg_level}
    # count_df_{agg_level}

    ## Generate final info using
    # raw_int_df --> Will be used in the output
    # global_flt_int_df
    # local_flt_int_df --> Will be used in the output --> If ignore_global_flt for a tag then directly starts from raw_int_df
    #

    """

    def __init__(self, query_obj, analytics_conf_path=None, settings_dict=None):
        """
        settings_dict = \
        {'output_settings':{
            'project_name': 'Jamo',
            'csv_output_agg_level': '15min',  #original, #shift, # shift_day, #24hours # [30min , 60min]
            'marker_agg_level': '15min', #original ## [30min , 60min], if original then uses the original time series
            'line_agg_level': '7D',  # Must be compatible with pandas time-series frequencies as given in the above link
            'ci_level': 0.95,
            'before_after': True},
        'default_settings': {
            'agg': {
                'levels': ['15min'],
                'ignore_global_filter': False,
                'min_records_ratio': 0.5}}}


        """
        self.q = query_obj
        self.groups = self.q.date_names

        assert not (analytics_conf_path is None and settings_dict is None), "Either analytics_conf_path or settings_dict should be provided"
        self.analytics_conf = AnalyticsConfig(analytics_conf_path=analytics_conf_path, settings_dict=settings_dict)

        blue = '#63a0cb'
        pink = '#ffb2c8'
        green = "#008000"
        red = '#ff0000'
        blue2 = "#b1b6fc"  # Original data
        self.visual_conf = VisualConfig(groups=self.groups, colors=['olive',blue,red,'DarkOliveGreen',green,red])
        if analytics_conf_path is not None:
            self._generate_local_flt_int_df()
        self._create_agg_dfs()

        # attribute to store local filters as boolean to show in removed graph
        # self.local_flt_boolean_dict = {}
        # Generate extra date related columns
        self.q.raw_int_df["month"] = self.q.raw_int_df.DateTime.dt.to_period("M").dt.to_timestamp().dt.tz_localize(self.q.raw_int_df.DateTime.dt.tz)
        self.q.raw_int_df["day"] = self.q.raw_int_df.DateTime.dt.to_period("D").dt.to_timestamp().dt.tz_localize(self.q.raw_int_df.DateTime.dt.tz)
        self.local_flt_int_df["month"] = self.local_flt_int_df.DateTime.dt.to_period("M").dt.to_timestamp().dt.tz_localize(self.local_flt_int_df.DateTime.dt.tz)
        self.local_flt_int_df["day"] = self.local_flt_int_df.DateTime.dt.to_period("D").dt.to_timestamp().dt.tz_localize(self.local_flt_int_df.DateTime.dt.tz)

        if self.q.global_flt_int_df is not None:
            self.q.global_flt_int_df["month"] = self.q.global_flt_int_df.DateTime.dt.to_period("M").dt.to_timestamp().dt.tz_localize(self.q.global_flt_int_df.DateTime.dt.tz)
            self.q.global_flt_int_df["day"] = self.q.global_flt_int_df.DateTime.dt.to_period("D").dt.to_timestamp().dt.tz_localize(self.q.global_flt_int_df.DateTime.dt.tz)

    def _generate_local_flt_int_df(self):
        """
        If ignore_global_flt then assign corresponding column from raw_int_df as initial version of local_flt_int_df
        """
        # Generate initial version of local_flt_int_df
        self.local_flt_int_df = self.q.global_flt_int_df.copy() if self.q.global_flt_int_df is not None  else self.q.raw_int_df.copy()  # Test this bit
        self.local_flt_int_df[self.analytics_conf.ignore_global_flt_tags] = self.q.raw_int_df[self.analytics_conf.ignore_global_flt_tags].copy()

        # Apply the actual local filters
        if self.analytics_conf.custom_settings is not None:
            for tag_name, props in self.analytics_conf.custom_settings.items():
                if props["local_filter"] is None:
                    continue
                self._apply_local_filter(tag_name, props["local_filter"])

    def _apply_local_filter(self, tag_name, filter_string):
        """
        filter_string: should be based on simplified tag names
        tag_name: simplified varsion of the tag_name --> as the function will be used internally

        # Accepts following operators: "<", ">", "=", "<=", ">=", "!="
        # OR a callable as a local filter --> Callable must accept a query obj and return a filtered query object
        # Apply filter string condition and get the remaining entries of tag_name for future use

        # Replaces filtered entries with NA in self.local_flt_int_df
        """
        operators_dict = {"<" : (lambda x, y: x < y), ">": (lambda x, y: x > y), "=": (lambda x, y: x == y),
                          "<=": (lambda x, y: x <= y), ">=": (lambda x, y: x >= y), "!=": (lambda x, y: x != y)}
        splitted_filters = [[simplifyTagNames(flt.split()[0].strip(), True), flt.split()[1].strip(), flt.split()[2].strip()] for flt in
                            filter_string.split(",")]
        final_flt = self.local_flt_int_df.DateTime.isna()
        for tag, operator, value in splitted_filters:
            # Logic here will be different than the preprocessors as filter will be applied on a different tag
            final_flt = final_flt | ~operators_dict[operator](self.local_flt_int_df[tag], pd.to_numeric(value, errors="ignore"))


        # self.local_flt_boolean_dict[tag_name] = final_flt
        self.local_flt_int_df.loc[final_flt, tag_name] = np.NAN

    def _create_agg_dfs(self):
        for agg_level in self.analytics_conf.default_settings["agg"]["levels"]:
            if agg_level == 'original':
                grouper = self.q.raw_int_df.DateTime
                threshold = int(grouper.reset_index().groupby("DateTime").count().max().values[0] * self.analytics_conf.default_settings["agg"]["min_records_ratio"])

            elif agg_level == 'shift':
                grouper = generate_shifts(self.q.raw_int_df)
                grouper = pd.Series(pd.to_datetime(grouper.values).tz_localize(self.q.raw_int_df.DateTime.dt.tz))
                grouper.name = "DateTime"
                grouper.index = self.q.raw_int_df.index

            elif agg_level == 'shift_day':
                grouper = generate_shift_day(self.q.raw_int_df)
                grouper = pd.Series(pd.to_datetime(grouper.values).tz_localize(self.q.raw_int_df.DateTime.dt.tz))
                grouper.name = "DateTime"
                grouper.index = self.q.raw_int_df.index
            else:
                grouper = self.q.raw_int_df.DateTime.dt.floor(freq=agg_level)

            threshold = int(grouper.reset_index().groupby("DateTime").count().max().values[0] * self.analytics_conf.default_settings["agg"]["min_records_ratio"])
            # Generate raw_agg_df
            tmp_cont_raw_agg_df = self.q.raw_int_df.groupby(grouper)[self.q.artifacts.cont].mean()
            tmp_categorical_raw_agg_df = self.q.raw_int_df.groupby(grouper)[self.q.artifacts.categorical].agg(lambda x: pd.Series.mode(x)[0])
            tmp_raw_agg_df = pd.concat([tmp_cont_raw_agg_df, tmp_categorical_raw_agg_df], axis=1)
            tmp_raw_agg_df = tmp_raw_agg_df[["g"] + sorted(
                [col for col in self.q.artifacts.categorical + self.q.artifacts.cont if col != "g"])].reset_index()


            # Generate flt_agg_df
            tmp_cont_flt_agg_df = self.local_flt_int_df.groupby(grouper)[self.q.artifacts.cont].mean()
            tmp_categorical_flt_agg_df = self.local_flt_int_df.groupby(grouper)[self.q.artifacts.categorical].agg(lambda x: x.value_counts().index[0])
            tmp_flt_agg_df = pd.concat([tmp_cont_flt_agg_df, tmp_categorical_flt_agg_df], axis=1)
            tmp_flt_agg_df = tmp_flt_agg_df[["g"] + sorted(
                [col for col in self.q.artifacts.categorical + self.q.artifacts.cont if col != "g"])].reset_index()

            # Generate cont_df
            tmp_count_df = self.local_flt_int_df.groupby(grouper).count()
            tmp_count_df = tmp_count_df[
                ["g"] + sorted([col for col in self.q.artifacts.categorical + self.q.artifacts.cont if col != "g"])]
            tmp_count_flt = (tmp_count_df >= threshold).reset_index()

            self.__setattr__(f"raw_agg_df_{agg_level}", tmp_raw_agg_df)
            self.__setattr__(f"flt_agg_df_{agg_level}", tmp_flt_agg_df)
            self.__setattr__(f"count_flt_{agg_level}", tmp_count_flt)

    def _get_extra_agg(self, tag_list, df_attr_name="raw_int_df", freq="5min", grouper=None):
        """
        - Don't use this method for any non-fixed frequency like month and check the results for ambiguous frequencies like 4days
        - Returns the aggregated version of the local_flt_int_df --> this would be identical to raw_int_df if no global & local filter applied
        - If local filter there it will be both

        df_attr_name: df to use for aggregation --> One of the following ["raw_int_df","global_flt_int_df", "local_flt_int_df"]
        """
        df = self.__getattr__(df_attr_name).copy()
        if grouper is None:
            grouper = df.DateTime.dt.floor(freq=freq)

        threshold = int(grouper.reset_index().groupby("DateTime").count().max().values[0] *
                        self.analytics_conf.default_settings["agg"]["min_records_ratio"])

        # Generate flt_agg_df
        tmp_cont_flt_agg_df = df.groupby(grouper)[[tag for tag in tag_list if tag in self.q.artifacts.cont]].mean()
        tmp_cat_flt_agg_df = df.groupby(grouper)[[tag for tag in tag_list if tag in self.q.artifacts.categorical]].agg(
            lambda x: x.value_counts().index[0])
        tmp_flt_agg_df = pd.concat([tmp_cont_flt_agg_df, tmp_cat_flt_agg_df], axis=1)
        tmp_flt_agg_df = tmp_flt_agg_df[["g"] + sorted(
            [col for col in self.q.artifacts.categorical + self.q.artifacts.cont if col != "g"])].reset_index()

        # Generate cont_df
        tmp_count_df = df.groupby(grouper).count()
        tmp_count_df = tmp_count_df[
            ["g"] + sorted([col for col in self.q.artifacts.categorical + self.q.artifacts.cont if col != "g"])]
        tmp_count_flt = (tmp_count_df >= threshold).reset_index()

        return tmp_flt_agg_df, tmp_count_flt

    def _generate_descriptive_table(self, tag_name, marker_agg_level):

        df_marker_retained = self.__getattribute__(f"flt_agg_df_{marker_agg_level}").loc[self.__getattribute__(f"count_flt_{marker_agg_level}")[tag_name],]
        df_marker_removed = self.__getattribute__(f"raw_agg_df_{marker_agg_level}").loc[self.__getattribute__(f"raw_agg_df_{marker_agg_level}").DateTime.isin(np.setdiff1d(self.__getattribute__(f"raw_agg_df_{marker_agg_level}").DateTime, df_marker_retained.DateTime))]

        stats_df_retained = pd.DataFrame(df_marker_retained.describe()).round(decimals=2)
        stats_df_removed = pd.DataFrame(df_marker_removed.describe()).round(decimals=2)

        fig = go.Figure()
        fig.add_trace(go.Table(
            header=dict(values=['', 'Retained', 'Removed']), #fill_color=['white', 'white','white']),
            cells=dict(values=[list(stats_df_retained.index), list(stats_df_retained[tag_name]),
                               list(stats_df_removed[tag_name])],
                       # fill_color=['white',blue,red])
        )))

        self.figs.append((fig, 3, 1, None))

    def _generate_ci_table(self, tag_name, marker_agg_level, confidence):

        df_marker_retained = self.__getattribute__(f"flt_agg_df_{marker_agg_level}").loc[self.__getattribute__(f"count_flt_{marker_agg_level}")[tag_name],]

        m = df_marker_retained[tag_name].mean()
        s = df_marker_retained[tag_name].std()
        dof = len(df_marker_retained[tag_name]) - 1

        # ci_mean
        t_crit = np.abs(t.ppf((1 - confidence) / 2, dof))
        mean_lix = round(m - s * t_crit / np.sqrt(len(df_marker_retained[tag_name])), 2)
        mean_uix = round(m + s * t_crit / np.sqrt(len(df_marker_retained[tag_name])), 2)
        ci_mean = (mean_lix, mean_uix)

        # ci_std
        std_lix = s * np.sqrt(dof / t.distributions.chi2.ppf((confidence) / 2, dof))
        std_uix = s * np.sqrt(dof / t.distributions.chi2.ppf((1 - confidence) / 2, dof))
        ci_std = (round(std_lix, 2), round(std_uix, 2))

        # ci_median
        dx = df_marker_retained[tag_name].sort_values(ascending=True, ignore_index=True)
        if len(dx) == 0:
            fig = go.Figure()
            self.figs.append((fig, 3, 2, None))
        else:
            factor = statistics.NormalDist().inv_cdf((1 + confidence) / 2)
            factor *= math.sqrt(len(dx))  # avoid doing computation twice
            med_lix = round(0.5 * (len(dx) - factor))
            med_uix = round(0.5 * (1 + len(dx) + factor))
            ci_median = (round(dx[med_lix], 2), round(dx[med_uix], 2))

            ci_df_main = pd.DataFrame([
                                           f'  mean: {str(ci_mean):.20s}',
                                           f'median: {str(ci_median):.20s}',
                                           f'   std: {str(ci_std):.20s}'],
                columns=[str(confidence) + ' confidence interval'])

            fig = go.Figure()
            fig.add_trace(go.Table(
                header=dict(values=['Confidence Intervals']),# fill_color=['white']),
                cells=dict(values=[list(ci_df_main.iloc[:, 0])]),# fill_color=[blue])
            ))

            self.figs.append((fig, 3, 2, None))

    def _generate_histogram(self, tag_name, agg_level):

        retained_df = self.__getattribute__(f"flt_agg_df_{agg_level}").loc[self.__getattribute__(f"count_flt_{agg_level}")[tag_name],]

        tmp = retained_df.groupby('g')[tag_name].apply(lambda x: x.values)
        data = tmp.values
        groups = list(tmp.index)

        colors = [self.visual_conf.meta_groups_to_color_mapper[(group.lower(), 'retained')] for group in groups]

        try:
            bin_size = min([(1 / 30) * (np.nanmax(group_data) - np.nanmin(group_data)) for group_data in data])
            data = [group_data[np.isfinite(group_data)] for group_data in data]
            fig = ff.create_distplot(data, group_labels=[group.capitalize() for group in groups], bin_size=bin_size, histnorm='probability', colors=colors )
            # fig.update_layout(show_legend=False)
        except:
            fig = go.Figure()

        self.figs.append((fig, 1, 1, None))

    def _generate_boxplot(self, tag_name, agg_level):

        relevant_filter_names = ['retained']
        relevant_meta_groups = list(itertools.product(self.groups, relevant_filter_names))

        retained_df = self.__getattribute__(f"flt_agg_df_{agg_level}").loc[self.__getattribute__(f"count_flt_{agg_level}")[tag_name],]

        fig = go.Figure()
        for idx, (group, filter_name) in enumerate(relevant_meta_groups[::-1]):
            tmp_df = retained_df.loc[retained_df.g == group,]
            fig.add_trace(go.Box(x=tmp_df.loc[:, tag_name],
                                 name=self.visual_conf.meta_groups_to_name_mapper[(group, filter_name)],
                                 marker_color=self.visual_conf.meta_groups_to_color_mapper[(group, filter_name)],
                                 line_color=self.visual_conf.meta_groups_to_color_mapper[(group, filter_name)],
                                 legendgroup=f"{group.capitalize()}", showlegend=False))

        self.figs.append((fig, 2, 1, None))

    def _generate_trend(self, tag_name, marker_agg_level="60min", line_agg_level="7D"):
        # If string column then just skip
        # if self.__getattribute__(f"raw_agg_df_{marker_agg_level}").dtypes[tag_name] == 'object':
        #     fig_dict = {f'fig_{group}': go.Figure() for group in self.groups}
        #     for idx, fig in enumerate(list(fig_dict.values())):
        #         self.figs.append((fig, 1, idx + 2, None))


        relevant_filter_names = ['retained', 'removed']
        relevant_meta_groups = list(itertools.product(self.groups, relevant_filter_names))

        df_marker_retained = self.__getattribute__(f"flt_agg_df_{marker_agg_level}").loc[self.__getattribute__(f"count_flt_{marker_agg_level}")[tag_name],]
        df_marker_removed = self.__getattribute__(f"raw_agg_df_{marker_agg_level}").loc[self.__getattribute__(f"raw_agg_df_{marker_agg_level}").DateTime.isin(np.setdiff1d(self.__getattribute__(f"raw_agg_df_{marker_agg_level}").DateTime, df_marker_retained.DateTime))]


        # Only for retained add smoothing lines
        #  use rolling 7 days
        rolling_mean_retained = pd.concat(
            [df_marker_retained.rolling(line_agg_level, on="DateTime")[[tag_name]].mean(), df_marker_retained.g],
            axis=1)

        fig_dict = {f'fig_{group}': go.Figure() for group in self.groups}

        for group, filter_name in relevant_meta_groups:
            if filter_name == "retained":
                tmp_df = df_marker_retained.loc[df_marker_retained.g == group,]
                fig_dict[f'fig_{group}'].add_trace(
                    go.Scatter(x=tmp_df["DateTime"], y=tmp_df[tag_name], legendgroup=f"{group.capitalize()}", legendgrouptitle_text=f"{group.capitalize()}",
                               name=f"Retained Data",
                               text=f"Retained Data",
                               mode='markers',
                               marker_color=self.visual_conf.meta_groups_to_color_mapper[(group, filter_name)]))
                fig_dict[f'fig_{group}'].add_trace(
                    go.Scatter(x=rolling_mean_retained.loc[rolling_mean_retained.g == group, "DateTime"],
                               y=rolling_mean_retained.loc[rolling_mean_retained.g == group, tag_name],
                               legendgroup=f"{group.capitalize()}", mode='lines',
                               name=f"{line_agg_level} Running Mean - Retained Data",
                               text=f"{line_agg_level} Running Mean",
                               line_color=self.visual_conf.meta_groups_to_color_mapper[(group, filter_name)]))

            if filter_name == "removed":
                tmp_df = df_marker_removed.loc[df_marker_removed.g == group,]
                fig_dict[f'fig_{group}'].add_trace(
                    go.Scatter(x=tmp_df["DateTime"], y=tmp_df[tag_name], name=f"Removed Data",
                               mode='markers', legendgroup=f"{group.capitalize()}",
                               marker_color=self.visual_conf.meta_groups_to_color_mapper[(group, filter_name)],
                               text=f"Removed Data"))

        y_range = [min(df_marker_retained[tag_name].min(), df_marker_removed[tag_name].min()),
                   max(df_marker_retained[tag_name].max(), df_marker_removed[tag_name].max())]
        y_range[0] = y_range[0] * 0.95 if y_range[0] > 0 else y_range[0] * 1.05
        y_range[1] = y_range[1] * 1.05 if y_range[1] > 0 else y_range[1] * 0.95

        for idx, fig in enumerate(list(fig_dict.values())):
            self.figs.append((fig, 1, idx + 2, y_range))

    def _generate_monthly_boxplot(self, tag_name, marker_agg_level):
        relevant_filter_names = ['retained']
        relevant_meta_groups = list(itertools.product(self.groups, relevant_filter_names))

        df_marker_retained = self.__getattribute__(f"flt_agg_df_{marker_agg_level}").loc[self.__getattribute__(f"count_flt_{marker_agg_level}")[tag_name],]

        df_marker_retained["year_month"] = df_marker_retained.DateTime.dt.to_period("M").dt.to_timestamp().dt.tz_localize(df_marker_retained.DateTime.dt.tz).map(lambda x: x.strftime('%Y-%m'))
        # df_marker_retained['year_month'] = df_marker_retained['month'].map(lambda x: x.strftime('%Y-%m'))

        fig_dict = {f'fig_{group}': go.Figure() for group in self.groups}

        for idx, (group, filter_name) in enumerate(relevant_meta_groups):
            tmp_df = df_marker_retained.loc[df_marker_retained.g == group,]
            fig_dict[f'fig_{group}'].add_trace(go.Box(y=tmp_df.loc[:, tag_name], x=tmp_df.loc[:, 'year_month'],
                                                      legendgroup=f"{group.capitalize()}", showlegend=False,
                                                      name=self.visual_conf.meta_groups_to_name_mapper[(group, filter_name)],
                                                      marker_color=self.visual_conf.meta_groups_to_color_mapper[(group, filter_name)],
                                                      line_color=self.visual_conf.meta_groups_to_color_mapper[(group, filter_name)]))


        y_range = [df_marker_retained[tag_name].min(), df_marker_retained[tag_name].max()]
        y_range[0] = y_range[0] * 0.95 if y_range[0] > 0 else y_range[0] * 1.05
        y_range[1] = y_range[1] * 1.05 if y_range[1] > 0 else y_range[1] * 0.95

        for idx, fig in enumerate(list(fig_dict.values())):
            self.figs.append((fig, 2, idx + 2, y_range))


    def _generate_before_after_table(self,tag_name, marker_agg_level, confidence=0.95):
        """
        Assumes 2 groups only
        """
        ## Mention in the output initial page that this test assumes all the other effecting factors were same for the 2 periods.
        retained_df = self.__getattribute__(f"flt_agg_df_{marker_agg_level}").loc[self.__getattribute__(f"count_flt_{marker_agg_level}")[tag_name],]
        retained_grouped = retained_df.groupby('g')[tag_name].apply(lambda x: x.values)
        # s_before = retained_grouped['before'] if 'before' in retained_grouped else np.ndarray(shape=0)
        # s_after = retained_grouped['after'] if 'after' in retained_grouped else np.ndarray(shape=0)
        try:
            s_before = retained_grouped[self.groups[0]]
            s_after = retained_grouped[self.groups[1]]
        except KeyError:
            logging.warning(f"{tag_name} does not have any entry in before OR after groups  -- skipping before/after table")
            fig = go.Figure()
            self.figs.append((fig, 3, 1, None))
            return
        if pd.Series(s_before).notna().sum() == 0 or pd.Series(s_after).notna().sum() == 0:
            fig = go.Figure()
            self.figs.append((fig, 3, 1, None))
            return

        out = ttest_ind(s_before, s_after, equal_var=True, nan_policy='omit', permutations=None, alternative='two-sided')

        n_before, u_before, var_before = s_before.shape[0], np.mean(s_before), np.var(s_before)
        n_after, u_after, var_after = s_after.shape[0], np.mean(s_after), np.var(s_after)
        var_pooled =  ((var_before*( n_before -1)) + (var_after * (n_after-1)))/(n_before + n_after -2 )
        u_diff = u_after - u_before

        ci = (round( u_diff- t.ppf(confidence,(n_before+n_after-2))*( var_pooled**0.5)*((1/n_after) + (1/n_before) )**0.5,2), round(u_diff + t.ppf(confidence,(n_before+n_after-2))*( var_pooled**0.5)*((1/n_after) + (1/n_before) )**0.5,2))
        p_value = round(out.pvalue,2)
        calc_test_stats = u_diff / ((var_pooled * ((1/n_after) + (1/n_before)))**0.5)
        calc_p_value = round(t.sf(np.abs(calc_test_stats), n_before + n_after - 2)*2,2) # Matching with out.pvalue -- just for sanity check after ci calculation which is based on manual calc.


        out_df = round(retained_df.groupby('g',sort=False)[tag_name].agg(['min','mean', 'median', 'max', 'std', 'count']).reset_index(),2)
        out_df[f'{confidence*100:.0f}% CI Diff Means'] = f"{ci}"
        out_df['P-value'] = f"{p_value*100:.1f}%"
        out_df = out_df.rename({"g":"Group"},axis=1)
        groups = out_df["Group"].values



        fig = go.Figure()
        fig.add_trace(go.Table(header=dict(values=[f"<b>{col}</b>" for col in out_df.columns],
                                           fill_color=["white"],), cells=dict(values=[out_df[k].tolist() for k in out_df.columns],
                                                                                    font_color=[['white','black']],
                                                                                    fill_color=[[self.visual_conf.meta_groups_to_color_mapper[(self.groups[0], 'retained')],self.visual_conf.meta_groups_to_color_mapper[(self.groups[1], 'retained')]]])))
        fig.update_layout(template="plotly_white")

        self.figs.append((fig, 3, 1, None))



    def _generate_plots_for_one_tag(self, tag_name, marker_agg_level, line_agg_level, ci_level):
        specs = [[{} for i in range(len(self.groups) + 1)],
                 [{} for i in range(len(self.groups) + 1)]]

        if len(self.groups) ==1:
            specs.append([{'type': 'table'}, {'type': 'table'}])
        elif len(self.groups) == 2:
            specs.append([{'type': 'table', 'colspan': 3}, None, None])

        one_tag_plots = make_subplots(rows=3, cols=len(self.groups) + 1, specs=specs,  column_widths=[0.5] + [0.5/len(self.groups)]*len(self.groups))
        self.figs = []
        self._generate_histogram(tag_name, agg_level=marker_agg_level)
        self._generate_trend(tag_name, marker_agg_level, line_agg_level)
        self._generate_boxplot(tag_name, agg_level=marker_agg_level)
        self._generate_monthly_boxplot(tag_name, marker_agg_level=marker_agg_level)
        if len(self.groups) ==2:
            self._generate_before_after_table(tag_name, marker_agg_level=marker_agg_level)
        elif len(self.groups) ==1:
            self._generate_descriptive_table(tag_name, marker_agg_level=marker_agg_level)
            self._generate_ci_table(tag_name, marker_agg_level=marker_agg_level , confidence=ci_level)

        for fig, row, col, y_range in self.figs:
            for t in fig.data:
                if col==1 and row==1:
                    t.update(showlegend=False)

                one_tag_plots.add_trace(t,row=row,col=col)

            if y_range is not None:
                one_tag_plots.update_yaxes(range=y_range, row=row, col=col, autorange=False)

        one_tag_plots.update_layout(
            xaxis=dict(nticks=20), xaxis2=dict(nticks=7),
            title_text='',
            title_x=0.15, template="plotly_white",
            autosize=False, width=2000, height=1250)
        # , yaxis2= { 'domain': [0.1, 0.5]}   , width=2000,   height=1200,
        # multiple_plot1.write_html('C:/Users/tavassoliny/Desktop/Projects/Con2_data/node1/'+var+'.html')

        # one_tag_plots.update_layout(showlegend=False)


        return one_tag_plots

    def save_df_as_csv(self, df_attr_name=None):
        if df_attr_name is None:
            df_attr_name = f"raw_agg_df_{self.analytics_conf.output_settings['marker_agg_level']}"
        savePresentableCSV(self.__getattribute__(df_attr_name).copy(), f"{df_attr_name}.csv", remove_g=True,
                           keep_index=False)

    def save_output_figures(self, tag_list, html_template_file_name='base.html', html_templates_dir="html_templates", project_name_extension=""):
        '''
        Only accepts numerical variables in tag_list
        '''

        tags = [{'tag' : tag, 'rolled_back_tag': self.q.artifacts.rolledback_tag_mapper[tag], 'desc': self.q.artifacts.description_mapper[tag], 'fig': self._generate_plots_for_one_tag(tag, marker_agg_level=self.analytics_conf.output_settings['marker_agg_level'], line_agg_level=self.analytics_conf.output_settings['line_agg_level'], ci_level=self.analytics_conf.output_settings["ci_level"]).to_html(full_html=False, include_plotlyjs='cdn')} for tag in tqdm(tag_list) if tag not in self.q.artifacts.categorical]

        dates = [{'start': start, 'end': end} for (start, end) in self.q.dates]
        dtypes = ['Interpolated Values'] if self.q.interpolated_values else []
        dtypes.append('Recorded Values') if self.q.recorded_values else None
        if self.q.global_filter is not None and self.q.global_filter != '':
            global_filters = self.q.global_filter.split(',')
        else:
            global_filters = ['']
        main_agg_level = {'24h':'24 hour', '12h':'12 hour', 'shift_day': 'Daily 7AM to 7AM', 'shift':'Shiftly', '60min': '1 hour', '30min': '30 minutes','15min':'15 minutes', '10min': '10 minutes', '5min': '5 minutes', 'original':'No Aggregation Applied'}[
            self.analytics_conf.output_settings['marker_agg_level']]

        file_loader = FileSystemLoader(html_templates_dir)
        env = Environment(loader=file_loader)
        template = env.get_template(html_template_file_name)
        output = template.render(project_name=self.analytics_conf.output_settings['project_name']+project_name_extension, tags=tags, dates=dates, dtypes=dtypes, global_filters=global_filters, main_agg_level=main_agg_level)

        # html_first_page = self.first_page_html()

        with open(f'{self.analytics_conf.output_settings["project_name"] + project_name_extension}.html', 'w') as f:
            f.write(output)
            f.close()



# TODO: Complete intro page
# TODO: Do final resizing on the final output
# TODO: Change colors and base on Yash's






class VisualConfig():
    def __init__(self, groups, filter_names=['original_data', 'retained', 'removed'],
                 colors=["olive", "black", "gold", "DarkOliveGreen", "darkgrey", "DarkGoldenRod"]):
        self.groups = groups

        # Extend colors to cover all combinations
        self.colors = colors * int(np.ceil((len(groups) * len(filter_names)) / len(colors)))
        self.meta_groups = list(itertools.product(groups, filter_names))
        self.meta_groups_to_color_mapper = dict(zip(self.meta_groups, self.colors))
        self.meta_groups_to_name_mapper = dict(zip(self.meta_groups, list(map(lambda x: x[0] + " - " + x[1] + ' data', self.meta_groups))))


class AnalyticsConfig:
    def __init__(self, analytics_conf_path=None, settings_dict=None):
        """
        settings_dict must have default_settings, output_settings just as in the analytics_config.yml files
        """
        if analytics_conf_path is None:
            assert settings_dict is not None , "Either analytics_conf_path or settings_dict should be provided"
            self.default_settings = settings_dict["default_settings"]
            self.output_settings = settings_dict["output_settings"]

        else:
            print("analytics config file path is provided -- Overriding dictionaries if provided")
            self.settings = load_yml(analytics_conf_path)
            self.default_settings = self.settings["default_settings"]
            self.custom_settings = self.settings["custom_settings"]
            self.output_settings = self.settings["output_settings"]
            del self.settings

            self.ignore_global_flt_tags = []
            if self.custom_settings is not None:
                self.ignore_global_flt_tags = [tag for tag, tag_settings_dict in self.custom_settings.items() if tag_settings_dict["agg"]["ignore_global_filter"]]


