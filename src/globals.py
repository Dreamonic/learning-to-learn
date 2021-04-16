import joblib

from tracker import Tracker, BaseTracker

cache = joblib.Memory(location='_cache', verbose=0)
trackers = {
    'default': BaseTracker(),
    'Training': Tracker.load('Training'),
    'ADAM': Tracker.load('ADAM'),
    'RMSprop': Tracker.load('RMSprop'),
    'SGD': Tracker.load('SGD'),
    'NAG': Tracker.load('NAG'),
    'LSTM': Tracker.load('LSTM'),
    'LSTM2Layer': Tracker.load('LSTM2Layer'),
    'LSTMBig': Tracker.load('LSTMBig'),
    'LSTMRelu': Tracker.load('LSTMRelu'),
    'TrainingBatch': Tracker.load('TrainingBatch')
}
