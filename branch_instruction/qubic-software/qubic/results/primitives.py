from abc import ABC, abstractmethod, abstractproperty
from numbers import Number
from qubic.sim.tools import twoscomp_to_signed
import numpy as np
import sys

class PrimitiveResult(ABC, np.lib.mixins.NDArrayOperatorsMixin):
    """
    Class template for primitive (raw) results; essentially a wrapper around np.ndarray. 
    Constructor accepts a bytes-like object (assumed to be read from FPGA memory), 
    which is then unpacked into a numpy.ndarray like object. Class has built-in support
    for basic array operations using __array_ufunc__ and __array_function__. Specific 
    ufunc or array_function implementations can be overriden as needed in child classes.
    np.lib.mixins.NDArrayOperatorsMixin is inherited to support python operators such 
    as +, *, etc.

    Child classes must implement:
        _unpack(self, data): convert binary data into a numpy array
        word_size(self): static method that returns the number of 32-bit words per datum

    """
    def __init__(self, data: bytes | np.ndarray, n_total_shots: int = None):
        if isinstance(data, bytes):
            self._array = self._unpack(data)
        else:
            self._array = data

    @abstractmethod
    def word_size(self):
        """
        size of a single datum in 32-bit words
        """
        pass

    @abstractmethod
    def _unpack(self, data: bytes) -> np.ndarray:
        pass

    def __array__(self) -> np.ndarray:
        return self._array

    @property
    def __array_interface__(self):
        return self._array.__array_interface__

    def __getitem__(self, index):
        return self._array[index]
    
    def __setitem__(self, index, item):
        self._array[index] = item
    
    def __len__(self):
        return len(self._array)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if 'out' in kwargs:
            assert isinstance(kwargs['out'], tuple)
            outtup = ()
            for i, out in enumerate(kwargs['out']):
                if isinstance(out, Number):
                    outtup += (out,)
                elif isinstance(out, self.__class__):
                    outtup += (out._array,)
                else:
                    return NotImplemented
            kwargs['out'] = outtup

        if method == '__call__':
            fnargs = []
            for input in inputs:
                if isinstance(input, Number):
                    fnargs.append(input)
                elif isinstance(input, self.__class__):
                    fnargs.append(self._array)
                else:
                    return NotImplemented
            result = ufunc(*fnargs, **kwargs)
            return self.__class__(result)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        args = [np.asarray(x) for x in args]
        return func(*args, **kwargs)

    @property
    def shape(self):
        return self._array.shape

    def __repr__(self):
        return f'{type(self).__name__}({self._array})'


class S11(PrimitiveResult):
    """
    Accumulated S11 data
    """

    def __init__(self, data: bytes | np.ndarray, n_total_shots=None):
        super().__init__(data, n_total_shots)
        if n_total_shots is not None:
            self._array = np.reshape(self._array, (n_total_shots, -1))

    @staticmethod
    def word_size():
        """
        Returns
        -------
        int:
            size of each sample in (unit is 32-bit word)
        """
        return 2

    def _unpack(self, data: bytes) -> np.ndarray:
        data = np.frombuffer(data, dtype=np.uint32)
        signed_data = np.reshape(twoscomp_to_signed(data.astype(int), nbits=32), (-1, 2))
        return 1j*signed_data[:, 0] + signed_data[:, 1]

    @property
    def real(self) -> np.ndarray:
        return self._array.real

    @property
    def imag(self) -> np.ndarray:
        return self._array.imag

class Sdbuf(PrimitiveResult):
    """
    Data that has been state-discriminated using QubiCML.
    """
    def __init__(self, data: bytes | np.ndarray, n_total_shots=None):
        super().__init__(data, n_total_shots)
        if n_total_shots is not None:
            self._array = np.reshape(self._array, (n_total_shots, -1))

    @staticmethod
    def word_size():
        return 1
    
    def _unpack(self, data: bytes) -> np.ndarray:
        data = np.frombuffer(data, dtype=np.int32)
        return (((data>>30)&0x3) + 1j*(data&0x3fffffff))
    
    
class U32(PrimitiveResult):
    """
    Generic uint32 type
    """
    def __init__(self, data: bytes | np.ndarray, n_total_shots=None):
        super().__init__(data, n_total_shots)
        if n_total_shots is not None:
            self._array = np.reshape(self._array, (n_total_shots, -1))

    @staticmethod
    def word_size():
        return 1

    def _unpack(self, data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.uint32)

def get_result_class(name):
    """
    Given a `name` (usually `dtype` field in `distproc.executable.ResultChannel`), return 
    the class representing that datatype. Converts from snake_case to CamelCase.
    """
    return getattr(sys.modules[__name__], ''.join(word.capitalize() for word in name.split('_')))

