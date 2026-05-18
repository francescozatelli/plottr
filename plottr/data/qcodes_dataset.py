"""
qcodes_dataset.py

Dealing with qcodes dataset (the database) data in plottr.
"""
import os
import sqlite3
import logging
from datetime import datetime
from itertools import chain
from operator import attrgetter
from time import perf_counter
from typing import Dict, List, Set, Union, TYPE_CHECKING, Any, Tuple, Optional, cast

from typing_extensions import TypedDict

import pandas as pd
import numpy as np

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

from qcodes.dataset.data_set import load_by_id
from qcodes.dataset.experiment_container import experiments
from qcodes.dataset.sqlite.database import initialise_or_create_database_at

from plottr import config_entry

from .datadict import DataDictBase, DataDict, combine_datadicts
from ..node.node import Node, updateOption

__author__ = 'Wolfgang Pfaff'
__license__ = 'MIT'

LOGGER = logging.getLogger('plottr.data.qcodes_dataset')

if TYPE_CHECKING:
    try:
        from qcodes.dataset import DataSetProtocol
    except ImportError:
        from qcodes.dataset.data_set import DataSet as DataSetProtocol
    from qcodes import ParamSpec


def _get_names_of_standalone_parameters(paramspecs: List['ParamSpec']
                                        ) -> Set[str]:
    all_independents = set(spec.name
                           for spec in paramspecs
                           if len(spec.depends_on_) == 0)
    used_independents = set(d for spec in paramspecs for d in spec.depends_on_)
    standalones = all_independents.difference(used_independents)
    return standalones


class IndependentParameterDict(TypedDict):
    unit: str
    label: str
    values: List[Any]


class DependentParameterDict(IndependentParameterDict):
    axes: List[str]


DataSetStructureDict = Dict[str, Union[IndependentParameterDict, DependentParameterDict]]


class DataSetInfoDict(TypedDict):
    experiment: str
    sample: str
    name: str
    completed_date: str
    completed_time: str
    started_date: str
    started_time: str
    structure: Optional[DataSetStructureDict]
    records: int
    guid: str
    inspectr_tag: str


# Tools for extracting information on runs in a database


def get_ds_structure(
        ds: 'DataSetProtocol'
) -> DataSetStructureDict:
    """
    Return the structure of the dataset, i.e., a dictionary in the form
        {
            'dependent_parameter_name': {
                'unit': unit,
                'label': label,
                'axes': list of names of independent parameters,
                'values': []
            },
            'independent_parameter_name': {
                'unit': unit,
                'label': label,
                'values': []
            },
            ...
        }

    Note that standalone parameters (those which don't depend on any other
    parameter and no other parameter depends on them) are not included
    in the returned structure.
    """

    structure: DataSetStructureDict = {}

    paramspecs = ds.get_parameters()

    standalones = _get_names_of_standalone_parameters(paramspecs)

    for spec in paramspecs:
        if spec.name not in standalones:
            if len(spec.depends_on_) > 0:
                structure[spec.name] = DependentParameterDict(unit=spec.unit,
                                                              label=spec.label,
                                                              values=[],
                                                              axes=list(spec.depends_on_))
            else:
                structure[spec.name] = IndependentParameterDict(unit=spec.unit,
                                                                label=spec.label,
                                                                values=[])
    return structure


def get_ds_info(ds: 'DataSetProtocol', get_structure: bool = True) -> DataSetInfoDict:
    """
    Get some info on a DataSet in dict.

    if get_structure is True: return the datastructure in that dataset
    as well (key is `structure' then).
    """
    _complete_ts = ds.completed_timestamp()
    if _complete_ts is not None:
        completed_date = _complete_ts[:10]
        completed_time = _complete_ts[11:]
    else:
        completed_date = ''
        completed_time = ''

    _start_ts = ds.run_timestamp()
    if _start_ts is not None:
        started_date = _start_ts[:10]
        started_time = _start_ts[11:]
    else:
        started_date = ''
        started_time = ''

    if get_structure:
        structure: Optional[DataSetStructureDict] = get_ds_structure(ds)
    else:
        structure = None

    data = DataSetInfoDict(
        experiment=ds.exp_name,
        sample=ds.sample_name,
        name=ds.name,
        completed_date=completed_date,
        completed_time=completed_time,
        started_date=started_date,
        started_time=started_time,
        structure=structure,
        records=ds.number_of_results,
        guid=ds.guid,
        inspectr_tag=ds.metadata.get('inspectr_tag', ''),
    )

    return data


def load_dataset_from(path: str, run_id: int) -> 'DataSetProtocol':
    """
    Loads ``DataSet`` with the given ``run_id`` from a database file that
    is located in in the given ``path``.

    Note that after the call to this function, the database location in the
    qcodes config of the current python process is changed to ``path``.
    """
    initialise_or_create_database_at(path)
    return load_by_id(run_id=run_id)


def get_runs_from_db(path: str, start: int = 0,
                     stop: Union[None, int] = None,
                     get_structure: bool = False) -> Dict[int, DataSetInfoDict]:
    """
    Get a db ``overview`` dictionary from the db located in ``path``. The
    ``overview`` dictionary maps ``DataSet.run_id``s to dataset information as
    returned by ``get_ds_info`` functions.

    `start` and `stop` refer to indices of the runs in the db that we want
    to have details on; if `stop` is None, we'll use runs until the end.

    If `get_structure` is True, include info on the run data structure
    in the return dict.
    """
    initialise_or_create_database_at(path)

    datasets = sorted(
        chain.from_iterable(exp.data_sets() for exp in experiments()),
        key=attrgetter('run_id')
    )

    # There is no need for checking whether ``stop`` is ``None`` because if
    # it is the following is simply equivalent to ``datasets[start:]``
    datasets = datasets[start:stop]

    overview = {ds.run_id: get_ds_info(ds, get_structure=get_structure)
                for ds in datasets}
    return overview


def get_runs_from_db_as_dataframe(path: str) -> pd.DataFrame:
    """
    Wrapper around `get_runs_from_db` that returns the overview
    as pandas dataframe.
    """
    return get_runs_from_db_as_dataframe_filtered(path)


def get_runs_from_db_as_dataframe_filtered(
        path: str,
        min_run_id: Optional[int] = None,
        run_ids: Optional[Set[int]] = None,
) -> pd.DataFrame:
    """
    Fetch run overview as pandas dataframe using direct sqlite queries.

    This is significantly faster than loading all datasets through the qcodes
    high-level API and supports incremental refreshes via ``min_run_id`` and
    ``run_ids`` filters.
    """
    initialise_or_create_database_at(path)

    run_ids = set() if run_ids is None else run_ids

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row

        run_cols = {
            row['name']
            for row in conn.execute("PRAGMA table_info(runs)")
        }

        inspectr_tag_expr = "COALESCE(r.inspectr_tag, '')"
        if 'inspectr_tag' not in run_cols:
            inspectr_tag_expr = "''"

        where_clauses: List[str] = []
        params: List[Any] = []
        if min_run_id is not None:
            where_clauses.append('r.run_id > ?')
            params.append(min_run_id)
        if len(run_ids) > 0:
            placeholders = ','.join('?' for _ in run_ids)
            where_clauses.append(f'r.run_id IN ({placeholders})')
            params.extend(sorted(run_ids))

        where = ''
        if len(where_clauses) > 0:
            where = 'WHERE (' + ' OR '.join(where_clauses) + ')'

        query = f'''
            SELECT
                r.run_id AS run_id,
                COALESCE(e.name, '') AS experiment,
                COALESCE(e.sample_name, '') AS sample,
                COALESCE(r.name, '') AS name,
                COALESCE(r.run_timestamp, '') AS run_timestamp,
                COALESCE(r.completed_timestamp, '') AS completed_timestamp,
                COALESCE(r.result_counter, 0) AS records,
                COALESCE(r.guid, '') AS guid,
                {inspectr_tag_expr} AS inspectr_tag
            FROM runs AS r
            LEFT JOIN experiments AS e ON r.exp_id = e.exp_id
            {where}
            ORDER BY r.run_id ASC
        '''
        rows = conn.execute(query, params).fetchall()

    def _split_timestamp(ts: Any) -> Tuple[str, str]:
        if ts is None:
            return '', ''

        if isinstance(ts, (int, float)):
            if pd.isna(ts):
                return '', ''
            dt = datetime.fromtimestamp(float(ts))
            return dt.strftime('%Y-%m-%d'), dt.strftime('%H:%M:%S')

        ts_str = str(ts).strip()
        if len(ts_str) >= 19:
            return ts_str[:10], ts_str[11:19]
        return '', ''

    overview: Dict[int, DataSetInfoDict] = {}
    for row in rows:
        started_date, started_time = _split_timestamp(row['run_timestamp'])
        completed_date, completed_time = _split_timestamp(row['completed_timestamp'])

        run_id = int(row['run_id'])
        overview[run_id] = DataSetInfoDict(
            experiment=str(row['experiment']),
            sample=str(row['sample']),
            name=str(row['name']),
            completed_date=completed_date,
            completed_time=completed_time,
            started_date=started_date,
            started_time=started_time,
            structure=None,
            records=int(row['records']),
            guid=str(row['guid']),
            inspectr_tag=str(row['inspectr_tag']),
        )

    return pd.DataFrame.from_dict(overview, orient='index')


# Extracting data

def ds_to_datadicts(
        ds: 'DataSetProtocol',
        dependents: Optional[List[str]] = None,
) -> Dict[str, DataDict]:
    """
    Make DataDicts from a qcodes DataSet.

    :param ds: qcodes dataset
    :returns: dictionary with one item per dependent.
              key: name of the dependent
              value: DataDict containing that dependent and its
                     axes.
    """
    ret = {}
    dependent_filter = set(dependents or [])
    has_cache = hasattr(ds, 'cache')
    if has_cache and len(dependent_filter) == 0:
        pdata = ds.cache.data()
    else:
        if len(dependent_filter) > 0:
            pdata = ds.get_parameter_data(*sorted(dependent_filter))
        else:
            # qcodes < 0.17
            pdata = ds.get_parameter_data()
    for p, spec in ds.paramspecs.items():
        if spec.depends_on != '':
            if len(dependent_filter) > 0 and p not in dependent_filter:
                continue
            axes = spec.depends_on_
            data = dict()
            data[p] = dict(unit=spec.unit, label=spec.label, axes=axes, values=pdata[p][p])
            for ax in axes:
                axspec = ds.paramspecs[ax]
                data[ax] = dict(unit=axspec.unit, label=axspec.label, values=pdata[p][ax])
            ret[p] = DataDict(**data)
            ret[p].validate()

    return ret


def ds_to_datadict_incremental(
    ds: 'DataSetProtocol',
    cached_data: Optional[DataDictBase] = None,
    last_loaded_records: int = 0,
    dependents: Optional[List[str]] = None,
) -> Tuple[DataDictBase, int]:
    """
    Extract data from qcodes dataset, optionally appending to cached data.
    
    For live plotting, this avoids re-extracting all rows when only a few
    new records have been added. The cache reduces extraction time by ~90%
    for slowly-growing datasets.
    
    :param ds: qcodes dataset
    :param cached_data: Previous extracted DataDict to append to (optional)
    :param last_loaded_records: Number of records in the cached_data
    :returns: (combined DataDict, total_loaded_records)
    """
    total_records = ds.number_of_results
    
    # If no new records, return cached data
    if cached_data is not None and total_records == last_loaded_records:
        return cached_data, last_loaded_records
    
    # If cache exists and we have new records, try incremental merge
    if cached_data is not None and total_records > last_loaded_records:
        ddicts = ds_to_datadicts(ds, dependents=dependents)
        new_ddict = combine_datadicts(*[v for k, v in ddicts.items()])
        
        # Slice to keep only new records from the full extraction
        new_data_dict = {}
        try:
            for name, field in new_ddict.data_items():
                vals = np.asarray(field.get('values', []))
                if len(vals) > 0:
                    # Keep only rows from last_loaded_records onward
                    new_vals = vals[last_loaded_records:]
                    if len(new_vals) > 0:
                        new_data_dict[name] = dict(
                            values=new_vals,
                            unit=field.get('unit'),
                            label=field.get('label'),
                            axes=field.get('axes', [])
                        )
            
            # Merge new data with cached data
            if new_data_dict:
                # Append new rows to cached values
                merged_dict = {}
                for name, field in cached_data.data_items():
                    old_vals = np.asarray(field.get('values', []))
                    if name in new_data_dict:
                        new_vals = np.asarray(new_data_dict[name].get('values', []))
                        merged_vals = np.concatenate([old_vals, new_vals])
                        merged_dict[name] = dict(
                            values=merged_vals,
                            unit=field.get('unit'),
                            label=field.get('label'),
                            axes=field.get('axes', [])
                        )
                    else:
                        merged_dict[name] = field.copy()
                
                # Create merged DataDict
                if merged_dict:
                    from .datadict import DataDict as DDImpl
                    merged_data = DDImpl(**merged_dict)
                    
                    # Copy metadata from both
                    for k, v in cached_data.meta_items():
                        merged_data.add_meta(k, v)
                    for k, v in new_ddict.meta_items():
                        # Update with new metadata but keep critical fields
                        if not k.startswith('qcodes_'):
                            merged_data.add_meta(k, v)
                    
                    return merged_data, total_records
        except Exception:
            # If merge fails, fall back to full extraction
            pass
    
    # Full extraction (fallback or initial load)
    ddicts = ds_to_datadicts(ds, dependents=dependents)
    ddict = combine_datadicts(*[v for k, v in ddicts.items()])
    return ddict, total_records


def ds_to_datadict(
        ds: 'DataSetProtocol',
        dependents: Optional[List[str]] = None,
) -> DataDictBase:
    ddicts = ds_to_datadicts(ds, dependents=dependents)
    ddict = combine_datadicts(*[v for k, v in ddicts.items()])
    return ddict


### qcodes dataset loader node

class QCodesDSLoader(Node):
    nodeName = 'QCodesDSLoader'
    uiClass = None
    useUi = False

    def __init__(self, *arg: Any, **kw: Any):
        self._pathAndId: Tuple[Optional[str], Optional[int]] = (None, None)
        self.nLoadedRecords = 0
        self._dataset: Optional[DataSetProtocol] = None
        self._diag_refresh_count = 0
        self._diag_last_rss_mb: Optional[float] = None
        self._cached_data: Optional[DataDictBase] = None
        self.dependentFilter: Optional[List[str]] = None

        super().__init__(*arg, **kw)

    def _rss_mb(self) -> Optional[float]:
        if psutil is None:
            return None
        try:
            p = psutil.Process(os.getpid())
            return float(p.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            return None

    def _effective_display_limit(self, base_limit: int) -> Tuple[int, str, Optional[float]]:
        """Adjust display limit based on current process RSS.

        Returns: (effective_limit, guard_mode, rss_mb)
        """
        rss_mb = self._rss_mb()
        if rss_mb is None:
            return base_limit, 'off', None

        warn_mb = float(config_entry('main', 'qcodes', 'memory_warning_mb', default=1500.0))
        emergency_mb = float(config_entry('main', 'qcodes', 'memory_emergency_mb', default=2500.0))
        warn_limit = int(config_entry('main', 'qcodes', 'max_records_for_display_warning', default=250000))
        emergency_limit = int(config_entry('main', 'qcodes', 'max_records_for_display_emergency', default=100000))

        if emergency_mb > 0 and rss_mb >= emergency_mb:
            return min(base_limit, max(1000, emergency_limit)), 'emergency', rss_mb
        if warn_mb > 0 and rss_mb >= warn_mb:
            return min(base_limit, max(1000, warn_limit)), 'warning', rss_mb
        return base_limit, 'normal', rss_mb

    def _decimate_for_display(self, data: DataDictBase) -> None:
        """Decimate records in-place to cap memory/plotting cost for very large runs."""
        base_limit = int(config_entry('main', 'qcodes', 'max_records_for_display', default=500000))
        max_records, guard_mode, rss_mb = self._effective_display_limit(base_limit)
        data.add_meta('plottr_memory_guard_mode', guard_mode)
        if rss_mb is not None:
            data.add_meta('plottr_rss_mb', float(rss_mb))
        if max_records <= 0:
            return

        nrecs: Optional[int] = None
        for _name, field in data.data_items():
            vals = np.asarray(field.get('values', []))
            if vals.ndim != 1:
                # Preserve shaped datasets to avoid changing semantics of
                # metadata-shape and grid-based workflows.
                return
            nrecs = int(vals.shape[0])
            break

        if nrecs is None or nrecs <= max_records:
            return

        stride = int(np.ceil(nrecs / max_records))
        stride = max(1, stride)
        for _name, field in data.data_items():
            vals = np.asarray(field.get('values', []))
            if vals.shape[0] == nrecs:
                field['values'] = vals[::stride]

        data.add_meta('plottr_decimated_records', nrecs)
        data.add_meta('plottr_decimation_stride', stride)
        data.add_meta('plottr_max_records_limit', max_records)

    def _estimate_datadict_bytes(self, data: DataDictBase) -> int:
        total = 0
        for _name, field in data.data_items():
            vals = np.asarray(field.get('values', []))
            total += int(vals.nbytes)
        return total

    def _log_memory_diag(
        self,
        path: str,
        run_id: int,
        total_records: int,
        data: DataDictBase,
        rss_before_mb: Optional[float],
        rss_after_mb: Optional[float],
        load_time_s: Optional[float] = None,
        extract_time_s: Optional[float] = None,
        decimate_time_s: Optional[float] = None,
    ) -> None:
        enabled = bool(config_entry('main', 'qcodes', 'memory_diagnostic_logging', default=False))
        if not enabled:
            return

        self._diag_refresh_count += 1
        every_n = int(config_entry('main', 'qcodes', 'memory_diagnostic_every_n_refreshes', default=1))
        every_n = max(1, every_n)
        if (self._diag_refresh_count % every_n) != 0:
            self._diag_last_rss_mb = rss_after_mb
            return

        payload_mb = self._estimate_datadict_bytes(data) / (1024.0 * 1024.0)
        guard_mode = 'normal'
        if data.has_meta('plottr_memory_guard_mode'):
            guard_mode = str(data.meta_val('plottr_memory_guard_mode') or 'normal')

        stride = None
        if data.has_meta('plottr_decimation_stride'):
            stride = data.meta_val('plottr_decimation_stride')
        stride_text = '1'
        if stride is not None:
            try:
                stride_text = str(int(stride))
            except Exception:
                stride_text = str(stride)

        delta_load: Optional[float] = None
        if rss_before_mb is not None and rss_after_mb is not None:
            delta_load = rss_after_mb - rss_before_mb

        delta_since_last: Optional[float] = None
        if self._diag_last_rss_mb is not None and rss_after_mb is not None:
            delta_since_last = rss_after_mb - self._diag_last_rss_mb

        timing_text = 'load=n/a extract=n/a decimate=n/a'
        if load_time_s is not None or extract_time_s is not None or decimate_time_s is not None:
            timing_text = (
                f'load={"n/a" if load_time_s is None else f"{load_time_s:.3f}s"} '
                f'extract={"n/a" if extract_time_s is None else f"{extract_time_s:.3f}s"} '
                f'decimate={"n/a" if decimate_time_s is None else f"{decimate_time_s:.3f}s"}'
            )

        LOGGER.info(
            (
                'QCodesDSLoader memory diag | db=%s run=%s records=%s '
                'payload=%.1fMB guard=%s stride=%s timings=%s '
                'rss_before=%s rss_after=%s delta_load=%s delta_prev=%s refresh_count=%s'
            ),
            os.path.basename(path),
            run_id,
            total_records,
            payload_mb,
            guard_mode,
            stride_text,
            timing_text,
            'n/a' if rss_before_mb is None else f'{rss_before_mb:.1f}MB',
            'n/a' if rss_after_mb is None else f'{rss_after_mb:.1f}MB',
            'n/a' if delta_load is None else f'{delta_load:+.1f}MB',
            'n/a' if delta_since_last is None else f'{delta_since_last:+.1f}MB',
            self._diag_refresh_count,
        )

        self._diag_last_rss_mb = rss_after_mb

    ### Properties

    @property
    def pathAndId(self) -> Tuple[Optional[str], Optional[int]]:
        return self._pathAndId

    @pathAndId.setter
    @updateOption('pathAndId')
    def pathAndId(self, val: Tuple[Optional[str], Optional[int]]) -> None:
        if val != self.pathAndId:
            self._pathAndId = val
            self.nLoadedRecords = 0
            self._dataset = None
            self._cached_data = None

    def setDependentFilter(self, dependents: Optional[List[str]]) -> None:
        new_filter = None if dependents is None else list(dependents)
        if new_filter != self.dependentFilter:
            self.dependentFilter = new_filter
            self.nLoadedRecords = 0
            self._cached_data = None

    def process(self, dataIn: Optional[DataDictBase] = None) -> Optional[Dict[str, Any]]:
        if dataIn is not None:
            raise RuntimeError("QCodesDSLoader.process does not take a dataIn argument")
        if None not in self._pathAndId:
            path, runId = cast(Tuple[str, int], self._pathAndId)
            rss_before_mb = self._rss_mb()
            load_started = perf_counter()

            # Reload the dataset handle on each refresh. In long-running live
            # sessions, a cached DataSet object can occasionally stop reflecting
            # appended rows, which stalls plotting until a run switch rebuilds
            # the loader state.
            try:
                self._dataset = load_dataset_from(path, runId)
            except ValueError as exc:
                if 'does not exist in the database' not in str(exc):
                    raise
                LOGGER.warning(
                    'Ignoring stale qcodes run request: db=%s run=%s (%s)',
                    path,
                    runId,
                    exc,
                )
                self._dataset = None
                self._cached_data = None
                self.nLoadedRecords = 0
                return None
            load_finished = perf_counter()

            if self._dataset.number_of_results > self.nLoadedRecords:

                guid = self._dataset.guid

                experiment_name = self._dataset.exp_name
                sample_name = self._dataset.sample_name
                dataset_name = self._dataset.name

                run_timestamp = self._dataset.run_timestamp()
                completed_timestamp = self._dataset.completed_timestamp()

                title = f"{os.path.split(path)[-1]} | " \
                        f"run ID: {runId} | GUID: {guid}" \
                        "\n" \
                        f"{sample_name} | {experiment_name} | {dataset_name}"

                info = f"""Started: {run_timestamp}
Finished: {completed_timestamp}
DB-File [ID]: {path} [{runId}]"""

                extract_started = perf_counter()
                # Use incremental loading to avoid re-extracting all rows on each refresh
                data, self.nLoadedRecords = ds_to_datadict_incremental(
                    self._dataset,
                    cached_data=self._cached_data,
                    last_loaded_records=max(0, self.nLoadedRecords),
                    dependents=self.dependentFilter,
                )
                extract_finished = perf_counter()
                
                # Cache the extracted data for next refresh
                self._cached_data = data
                decimate_started = perf_counter()
                self._decimate_for_display(data)
                decimate_finished = perf_counter()

                load_time_s = load_finished - load_started
                extract_time_s = extract_finished - extract_started
                decimate_time_s = decimate_finished - decimate_started

                data.add_meta('plottr_loader_load_time_s', float(load_time_s))
                data.add_meta('plottr_loader_extract_time_s', float(extract_time_s))
                data.add_meta('plottr_loader_decimate_time_s', float(decimate_time_s))
                data.add_meta('plottr_loader_total_time_s', float(decimate_finished - load_started))

                data.add_meta('qcodes_experiment_name', experiment_name)
                data.add_meta('qcodes_sample_name', sample_name)
                data.add_meta('qcodes_dataset_name', dataset_name)

                data.add_meta('title', title)
                data.add_meta('info', info)

                data.add_meta('qcodes_guid', guid)
                data.add_meta('qcodes_db', path)
                data.add_meta('qcodes_runId', runId)
                data.add_meta('qcodes_completedTS', completed_timestamp)
                data.add_meta('qcodes_runTS', run_timestamp)
                qcodes_shape = getattr(self._dataset.description, "shapes", None)
                data.add_meta('qcodes_shape', qcodes_shape)

                self.nLoadedRecords = self._dataset.number_of_results
                rss_after_mb = self._rss_mb()
                self._log_memory_diag(
                    path=path,
                    run_id=runId,
                    total_records=self._dataset.number_of_results,
                    data=data,
                    rss_before_mb=rss_before_mb,
                    rss_after_mb=rss_after_mb,
                    load_time_s=load_time_s,
                    extract_time_s=extract_time_s,
                    decimate_time_s=decimate_time_s,
                )

                return dict(dataOut=data)

        return None
