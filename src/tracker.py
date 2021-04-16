import glob
import json
import os
import sys
import time
import uuid

from util import file_path_to_name


class BaseTracker:
    def forward(self, net, inp, out):
        pass

    def start_timer(self, name):
        pass

    def stop_timer(self, name):
        pass

    def save(self):
        pass


class Tracker(BaseTracker):
    cache_location = '_tracker_cache'

    def __init__(self, name, queries_size=0, timings=None):
        self.name = name
        # if queries is None:
        #     queries = []
        # self._queries = queries
        self._queries_size = queries_size

        if timings is None:
            timings = {}
        self._timings = timings
        self._start_times = {}

    def forward(self, net, inp, out):
        self.track_query(inp, out)

    def track_query(self, query, output):
        # self._queries.append((query, output))
        self._queries_size += 1

    def start_timer(self, name):
        self._start_times[name] = time.perf_counter()

    def stop_timer(self, name):
        if name not in self._start_times:
            raise AssertionError(f"Cannot stop the timer called: {name}, have you called start_timer({name})?")
        self._timings[name] = time.perf_counter() - self._start_times[name] + self._timings.get(name, 0)
        del self._start_times[name]

    @property
    def size(self):
        return self._queries_size

    def reset(self):
        # self._queries = []
        self._queries_size = 0

        self._timings = {}
        self._start_times = {}

    def save(self, path=''):
        encoded = {
            'name': self.name,
            # 'query': self._queries,
            'query_size': self._queries_size,
            'timings': self._timings,
        }
        main_file = file_path_to_name(path)
        base_path = os.path.join(Tracker.cache_location, main_file)
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        with open(os.path.join(Tracker.cache_location, main_file, f'{self.name}_{uuid.uuid4()}.json'), 'w+') as fp:
            json.dump(encoded, fp)

    @property
    def timings(self):
        return self._timings

    @staticmethod
    def load(name, path=''):
        main_file = file_path_to_name(path)
        for path in glob.glob(os.path.join(Tracker.cache_location, main_file, f'{name}*')):
            with open(f'{path}', 'r+') as fp:
                dic = json.load(fp)
                return Tracker(
                    dic.get('name'),
                    # dic.get('query'),
                    dic.get('query_size', 0),
                    dic.get('timings'),
                )
        return Tracker(name)
