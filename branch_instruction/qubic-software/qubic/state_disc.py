"""
TODO: maybe add an abstract state disc class/interface. 
"""
from sklearn import mixture
import numpy as np
import pickle as pkl
from qubic.rfsoc.hwconfig import ChannelConfig
from typing import Dict, List
from qubic.results.primitives import S11
import logging
import time
import json
import os


class GMMStateDiscriminator:
    """
    Class for single-qudit state discrimination using a Gaussian-mixture model (GMM).
    Collections of state-discriminators (across multiple qubits) are managed using
    the GMMManager class
    """
    def __init__(self, n_states: int = 2, fit_iqdata: np.ndarray = None, load_dict: dict = None):
        self.labels = np.array(n_states*[np.nan])
        self.n_states = n_states
        self.fit_iqpoints = np.empty((0,), dtype=np.complex128)
        self.gmmfit = None
        if fit_iqdata is not None:
            self.fit(fit_iqdata)
#        self.fitdatetime=None            
        if load_dict is not None:
            self.loadfromdict(load_dict)

    def fit(self, iqdata: S11, update: bool = True):
        """
        Fit GMM model (determine blob locations and uncertainties) based
        on input iqdata.

        Parameters
        ----------
        iqdata: np.ndarray
            array of complex-valued IQ shots
        update: bool
            if True (default), then update existing model with new 
            data, else re-create model using only new data for fit
        """
        iqdata = np.asarray(iqdata)
        if update:
            self.fit_iqpoints = np.append(self.fit_iqpoints, iqdata)
        else:
            self.fit_iqpionts = iqdata

        self.gmmfit = mixture.GaussianMixture(self.n_states
                #                ,means_init=((-60000,-60000),(100000,100000))
                )
        nanmask = np.isnan(self.fit_iqpoints)
        if np.any(nanmask):
            logging.getLogger(__name__).warning('GMM fit data contains NaNs, ignoring')
        fit_points = self.fit_iqpoints[~nanmask]
        self.gmmfit.fit(self._format_complex_data(fit_points))
        self.fitdatetime=time.strftime('%Y%m%d_%H%M%S_%Z')

    def predict(self, iqdata: S11, use_label=True) -> np.ndarray:
        """
        Label iqdata with qubit state as determined by 

        Parameters
        ----------
        iqdata: np.ndarray
            array of complex-valued IQ shots

        Returns
        -------
        np.ndarray
            array of labeled data, corresponding to self.labels; same
            shape as iqdata
        """
        iqdata = np.asarray(iqdata)
        nanmask = np.isnan(iqdata)
        pred_iqdata = iqdata.copy()
        pred_iqdata[nanmask] = 0
        predictions = self.gmmfit.predict(self._format_complex_data(pred_iqdata))
        #ipdb.set_trace()
        if use_label:
            predictions = self.labels[predictions]
        else:
            predictions = predictions.astype(np.float64)
        predictions = np.reshape(predictions, iqdata.shape)
        predictions[nanmask] = np.nan
        return predictions

    def _format_complex_data(self, data: np.ndarray):
        return np.vstack((np.real(data.flatten()), np.imag(data.flatten()))).T

    def set_labels(self, labels: list[int | str] | np.ndarray):
        """
        Set all labels according to provided list
        """
        if len(labels) != self.n_states:
            raise Exception('Must have {} labels!'.format(self.n_states))
        self.labels = np.asarray(labels)

    def get_threshold_angle(self, label0: str | int = 0, label1: str | int = 1):
        """
        Get the angle (wrt to horizontal) of the midpoint between two labels in the 
        IQ plane.

        Parameters
        ----------
        label0: str | int
        label1: str | int

        Returns
        -------
        float: threshold angle in radians
        """
        blob0_coords = self.gmmfit.means_[self.labels==label0][0]
        blob1_coords = self.gmmfit.means_[self.labels==label1][0]
        threshpoint = (blob0_coords + blob1_coords)/2
        return np.arctan2(threshpoint[1], threshpoint[0])

    def switch_labels(self):
        """
        Switch 1 and 0 labels. For higher energy states, reverse the order of
        the labels array.
        """
        self.labels = self.labels[::-1]

    def set_none_label(self, label: int | str):
        """
        If any single label is None, set it to `label`
        """
        nonesum=sum(self.labels==None)
        if nonesum==1:
            index=np.where(self.labels==None)[0][0]
            self.labels[index]=label

    def set_labels_maxtomin(self, iqdata: S11, labels_maxtomin: list | np.ndarray = [0,1]):
        """
        Set labels in descending order based on number of shots in a given blob.
        e.g. if labels_maxtomin = [0,1], this function will assign label 0
        to the GMM blob with the highest population in iqdata, and 1 to the next
        highest. If any rank-ordered blob should have unchanged assignment, set 
        to None. (e.g. labels_maxtomin=[None, 1] will only assign 1 to the lowest
        population blob)

        Parameters
        ----------
        iqdata: np.ndarray
            raw complex IQ shots
        labels_maxtomin: list or np.ndarray
            order of labels to assign, in descending order
            of prevelance in iqdata
        """
        iqdata = np.asarray(iqdata)
        assert len(labels_maxtomin) <= self.n_states
        pred = self.predict(iqdata, use_label=False)
        n_pred = [] #number of shots at label index 0, 1, 2, etc
        for i in range(self.n_states):
            n_pred.append(np.nansum(pred == i))

        blobinds_sorted = np.argsort(n_pred)[::-1] #sort blobinds in order of max prevelance
        for i, label in enumerate(labels_maxtomin):
            if label is not None:
                self.labels[blobinds_sorted[i]] = label
                logging.getLogger(__name__).debug('set label ', blobinds_sorted[i], label)

    def dict_serialize(self):
        gmmdictser={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in self.gmmfit.__dict__.items()}
        dictout=dict(labels=self.labels.tolist())
        if hasattr(self,'fitdatetime'):
            dictout.update(dict(fitdatetime=self.fitdatetime))
        dictout.update(dict(gmm=gmmdictser))
        return dictout

    def loadfromdict(self, dictin: dict):
        """
        Load GMM model (labels + means for each state) from a dictionary
        """
        gmmdictser={k:np.array(v) if isinstance(v,list) else v for k,v in dictin.items()}
        if 'labels' in dictin:
            self.labels=np.array(dictin['labels'])
        else:
            self.labels=None
        if 'fitdatetime' in dictin:
            self.fitdatetime=dictin['fitdatetime']
        else:
            self.fitdatetime=None
        if 'gmm' in dictin:
            self.gmmfit = mixture.GaussianMixture()
            for k,v in dictin['gmm'].items():
                setattr(self.gmmfit,k,np.array(v) if isinstance(v,list) else v)
            self.n_states=self.gmmfit.n_components

class GMMManager:
    """
    Class for managing multi-qubit GMM classifiers. 

    Attributes
    ----------
    chan_to_qubit: dict
        map from hardware channel (usually core_ind) to qubitid
    gmm_dict: dict
        dictionary of GMMStateDiscriminator objects. keys are qubitid

    """

    def __new__(cls,
                load_file: str = None, 
                gmm_dict: Dict[str, GMMStateDiscriminator] = None, 
                chanmap_or_chan_cfgs: Dict[int, str] | Dict[str, ChannelConfig] = None,
                load_json: str = None,
                n_states: int = 2):
        """
        Must specify either load_file, or chanmap_or_chan_cfgs. If load_file is NOT
        specified, can specify gmm_dict to load in existing set of GMM models.

        Parameters
        ----------
        load_file: str
            If provided, loads GMM manager object from pkl filename
        gmm_dict: dict
            Existing GMM dictionary, indexed by qubit. Loads this into
            the object
        chanmap_or_chan_cfgs: dict
            dict of ChannelConfig objects, or dictionary mapping 
            channels to qubits. 
        load_json: str
            If provided, loads GMM manager object from json filename
        n_states: int
            Number of states to classify
            
        """
        if load_file is not None:
            assert gmm_dict is None
            with open(load_file, 'rb') as f:
                inst = pkl.load(f)
            if chanmap_or_chan_cfgs is not None:
                inst._resolve_chanmap(chanmap_or_chan_cfgs)
            return inst
        else:
            return super(GMMManager, cls).__new__(cls)

    def __init__(self, 
                 load_file: str = None, 
                 gmm_dict: Dict[str, GMMStateDiscriminator] = None, 
                 chanmap_or_chan_cfgs: Dict[int, str] | Dict[str, ChannelConfig] = None,
                 load_json: str = None,
                 n_states: int = 2):
        """
        Must specify either load_file, or chanmap_or_chan_cfgs. If load_file is NOT
        specified, can specify gmm_dict to load in existing set of GMM models.

        Parameters
        ----------
        load_file: str
            If provided, loads GMM manager object from pkl filename
        gmm_dict: dict
            Existing GMM dictionary, indexed by qubit. Loads this into
            the object
        chanmap_or_chan_cfgs: dict
            dict of ChannelConfig objects, or dictionary mapping 
            channels to qubits. 
        load_json: str
            If provided, loads GMM manager object from json filename
        n_states: int
            Number of states to classify
        """
        self.n_states=n_states
        if gmm_dict is not None:
            assert isinstance(gmm_dict, dict)
            assert load_file is None
            self.gmm_dict = gmm_dict
            assert chanmap_or_chan_cfgs is not None
            self._resolve_chanmap(chanmap_or_chan_cfgs)
        elif load_file is not None: #object was loaded from pkl
            assert gmm_dict is None
            if chanmap_or_chan_cfgs is not None:
                self._resolve_chanmap(chanmap_or_chan_cfgs)
        elif load_json is not None:
            self.gmm_dict = {}
            with open(load_json) as jfile:
                jdict=json.load(jfile)
                for k,v in jdict.items():
                    self.gmm_dict[k]=GMMStateDiscriminator(load_dict=v)
        else:
            self.gmm_dict = {}
            assert chanmap_or_chan_cfgs is not None
            self._resolve_chanmap(chanmap_or_chan_cfgs)

    def update(self, gmm_manager):
        assert isinstance(gmm_manager, GMMManager)
        self.gmm_dict.update(gmm_manager.gmm_dict)

    def _resolve_chanmap(self, chanmap_or_chan_cfgs):
        if isinstance(list(chanmap_or_chan_cfgs.values())[1], str):
            # this is a channel to qubit map
            self.chan_to_qubit = chanmap_or_chan_cfgs
        else:
            # this is a chan cfg dict
            self.chan_to_qubit = {dest: dest.split('.')[0] 
                                  for dest, channel in chanmap_or_chan_cfgs.items() 
                                  if isinstance(channel, ChannelConfig) and channel.acc_mem_name is not None}

    def fit(self, iq_shot_dict: Dict[str, S11]):
        """
        Fit GMM models based on input data in iq_shot_dict. If model doesn't exist, create it,
        if so, update existing model with new data.

        Parameters
        ----------
        iq_shot_dict: Dict[str, np.ndarray]
            dictionary of IQ data, keyed by str(channel_number), or qubit
            
        """
        for chan, iq_shots in iq_shot_dict.items():
            if self._get_gmm_key(chan) in self.gmm_dict.keys():
                self.gmm_dict[self._get_gmm_key(chan)].fit(iq_shots)
            else:
                self.gmm_dict[self._get_gmm_key(chan)] = GMMStateDiscriminator(fit_iqdata=iq_shots, n_states=self.n_states)

    def _get_gmm_key(self, chan: str) -> str:
        """
        Checks if `chan` is a qubit or numbered channel (as a string). If qubit,
        returns `chan`, else converts to corresponding qubit.
        """
        if chan in list(self.chan_to_qubit.values()):
            return chan
        else:
            return self.chan_to_qubit[chan]

    def get_threshold_angle(self, qubit: str, label0: int| str = 0, label1: int | str = 1) -> float:
        """
        Get the threshold angle for a particular qubit; wrapper around 
        GMMStateDiscriminator.get_threshold_angle

        Parameters
        ----------
        qubit: str
        label0: int | str
        label1: int | str

        Returns
        -------
        float:
            angle in radians

        """
        return self.gmm_dict[qubit].get_threshold_angle(label0, label1)

    def predict(self, iq_shot_dict: Dict[str, S11], output_keys: str = 'qubit') -> Dict[str, np.ndarray]:
        """
        Assign labels to IQ shots.

        Parameters
        ----------
        iq_shot_dict : dict
            keys: channel no. or qubitid
            values: complex array of shots to predict
        output_keys : str
            either 'qubit' or 'channel'

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing arrays of labeled data, corresponding to self.labels; same
            shape as iqdata; keyed by qubit or channel, depending on `output_keys`
        """
        result_dict = {}
        for chan, iq_shots in iq_shot_dict.items():
            result = self.gmm_dict[self._get_gmm_key(chan)].predict(iq_shots)
            if output_keys == 'qubit':
                result_dict[self._get_gmm_key(chan)] = result
            elif output_keys == 'channel':
                if chan in self.chan_to_qubit.keys():
                    result_dict[chan] = result
                else:
                    raise NotImplementedError
            else:
                raise ValueError('output_keys must be qubit or channel')

        return result_dict

    def set_labels_maxtomin(self, iq_shot_dict: Dict[str, S11], labels_maxtomin: list):
        """
        Batched version of GMMStateDiscriminator.set_labels_maxtomin

        Parameters
        ----------
        iq_shot_dict : dict
            Set of complex IQ values
        labels_maxtomin : list
            Labels to assign in descending order of prevelance
        """
        for chan, iq_shots in iq_shot_dict.items():
            self.gmm_dict[self._get_gmm_key(chan)].set_labels_maxtomin(iq_shots, labels_maxtomin)

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pkl.dump(self, f)

    def savejson(self, filename: str,update: bool = True, indent: int = 4):
        """
        Serialize into dictionary and save as json.
        """
        newdict={k:v.dict_serialize() for k,v in self.gmm_dict.items()}
        if update:            
            if os.path.isfile(filename):
                with open(filename) as f:
                    serdict=json.load(f)
            else:
                serdict={}
            serdict.update(newdict) 
        else:
            serdict=newdict
        with open(filename, 'w') as f:
            json.dump(serdict,f,indent=indent)
                
