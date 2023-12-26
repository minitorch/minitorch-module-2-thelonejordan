from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, List, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage
    """
    return int(sum(i * s for i, s in zip(index, strides)))


def to_index(
    ordinal: int, shape: Shape, out_index: OutIndex, strides: Optional[Strides] = None
) -> None:
    """
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    # donot reassign out_index, you can assign to it
    if not isinstance(shape, np.ndarray):
        shape = np.array(shape, dtype=np.int32)
    dim, size = shape.size, int(prod(shape))
    assert len(out_index) == dim, "out_index"  # TODO: convert to out_index.size
    if ordinal < 0 or ordinal >= size:
        raise IndexingError("Ordinal position out of bounds")
    strides = np.array(strides_from_shape(tuple(shape))) if strides is None else strides
    if not isinstance(strides, np.ndarray):
        strides = np.array(strides, dtype=np.int32)
    assert strides.size == dim, "strides and shape dims don't match"
    for i in reversed(np.argsort(np.multiply(strides, shape)).tolist()):
        stride = strides[i]
        out_index[i] = ordinal // stride
        ordinal %= stride


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None
    """
    # donot reassign out_index, you can assign to it
    if not isinstance(big_index, np.ndarray):
        big_index = np.array(big_index, dtype=np.int32)
    if not isinstance(big_shape, np.ndarray):
        big_shape = np.array(big_shape, dtype=np.int32)
    if not isinstance(shape, np.ndarray):
        shape = np.array(shape, dtype=np.int32)
    dim, bdim = shape.size, big_shape.size
    assert len(out_index) == dim  # TODO: convert to out_index.size
    assert big_index.size == bdim
    for i in range(dim):
        x, y = dim - i - 1, bdim - i - 1
        if shape[x] == big_shape[y]:
            out_index[x] = big_index[y]
        elif shape[x] == 1:
            out_index[x] = 0
        else:
            raise NotImplementedError("broadcasting edge case encountered!")


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    if shape1 == (1,):
        return shape2
    if shape2 == (1,):
        return shape1
    ndim1, ndim2 = len(shape1), len(shape2)
    ndim = max(ndim1, ndim2)
    shape = [0 for _ in range(ndim)]
    for i in range(min(ndim1, ndim2)):
        if (
            (sh2 := shape2[ndim2 - i - 1]) == (sh1 := shape1[ndim1 - i - 1])
            or sh1 == 1
            or sh2 == 1
        ):
            shape[lidx := (ndim - i - 1)] = max(sh1, sh2)
            continue
        raise IndexingError("shapes not broadcastable")
    shape[:lidx] = (shape1 if ndim == ndim1 else shape2)[:lidx]
    return tuple(shape)


def strides_from_shape(shape: UserShape) -> UserStrides:
    dim = len(shape)
    size = int(prod(shape))
    strides = [0] * dim
    for i in range(dim):
        size //= shape[i]
        strides[i] = size
    return tuple(strides)


def indices(shape: Tuple[int, ...]) -> Iterable[Sequence[int]]:
    def addDim(idx: int) -> Sequence[List[int]]:
        if idx == 1:
            return [[i] for i in range(shape[idx - 1])]
        else:
            ret = []
            for i in range(shape[idx - 1]):
                for x in addDim(idx - 1):
                    x.append(i)
                    ret.append(x)
            return ret

    for idx in addDim(len(shape)):
        yield tuple(idx)


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)
        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(aindex, self._strides)

    def indices(self) -> Iterable[UserIndex]:
        return indices(tuple(self.shape))

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            order (list): a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # Create a new TensorData with the same storage
        return TensorData(
            self._storage.copy(),
            tuple(self.shape[int(i)] for i in order),
            tuple(self.strides[int(i)] for i in order),
        )

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            v = self.get(tuple(index))
            s += l + f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
