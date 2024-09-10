from __future__ import annotations

from abc import ABC, abstractmethod
from ctypes import Structure, addressof, memmove, sizeof, c_uint8
from itertools import chain
from queue import Queue
from threading import Event, Lock, Thread
import warnings

from hephaistos import (
    Command,
    Program,
    RawBuffer,
    Submission,
    Subroutine,
    Tensor,
    Timeline,
    beginSequence,
    createSubroutine,
    retrieveTensor,
    updateTensor,
)
from hephaistos.util import StructureTensor
from numpy import frombuffer, float32

from numpy.typing import DTypeLike, NDArray
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)


class PipelineStage(ABC):
    """
    Base class for pipeline stages.

    Handles double buffering of configuration, allowing the CPU to update one
    configuration while the other is still in use by the GPU, prevents the GPU
    in the best case from waiting on the CPU, thus boosting throughput.

    UBOs can be defined by passing a dictionary mapping binding names to a
    ctypes structure describing it. It's field name are reflected, added to the
    stages parameters and can be accessed via `getParam` / `setParam`. If they
    start with an underscore, they are considered private and won't show up in
    the `fields` property and output of `getParams` and print.

    Additional properties that should be able to be set via `getParam` /
    `setParam` can be passed by name in the `extras` set. They are most likely
    implemented as properties.

    During creation of a `Pipeline` the method `run(i)` will be called for both
    buffers only once each. It must return a list of commands that define the
    function of the stage and will be used to create a pipeline subroutine that
    gets reused.

    The current idle configuration get's updated via a call to `update(i)`,
    which writes the current local state, i.e. the one defined by the Python
    object, to device memory. Any private parameters can be updated by
    overloading the `_finishParams(i)` method, which gets called during
    `update(i)` before the actual write to the device.

    Parameters
    ----------
    params: {str: Structure}, default={}
        Dictionary of named ctype structure containing the stage's parameters.
        Each structure will be allocated on the CPU side and twice on the GPU
        for buffering. The latter can be bound in programs
    extra: {str}, default={}
        Set of extra parameter name, that can be set and retrieved using the
        stage api. Take precedence over parameters defined by structs. Should
        be be implemented in subclasses as properties.
    """

    name = "stage"
    """default stage name. Should be changed in subclasses."""

    def __init__(
        self, params: Dict[str, Type[Structure]] = {}, extra: Set[str] = set()
    ) -> None:
        # create local configuration
        self._local = {name: param() for name, param in params.items()}
        # create double buffered device config
        self._device = [
            {name: StructureTensor(param, True) for name, param in params.items()}
            for _ in range(2)
        ]
        # check tensors are mapped
        if any(
            not tensor.isMapped for buffer in self._device for tensor in buffer.values()
        ):
            raise RuntimeError("PipelineStage requires support for mapped tensors!")
        # create params map name -> param struct
        self._params = {
            fieldName: fieldType.from_address(
                addressof(self._local[paramName])
                + getattr(params[paramName], fieldName).offset
            )
            for paramName, param in params.items()
            for fieldName, fieldType in param._fields_
            # if not fieldName.startswith("_")
        }
        self._extra = extra
        # collect all fields without private one (starts with "_")
        self._fields = frozenset(extra | self._params.keys())
        self._public = frozenset(f for f in self._fields if not f.startswith("_"))

    @property
    def fields(self) -> Set[str]:
        """Set of all public parameter names"""
        return self._public

    def __dir__(self) -> Iterable[str]:
        return chain(super().__dir__(), self._public)

    def getParam(self, name: str) -> Any:
        """Returns the parameter specified by its name"""
        if name in self._extra:
            return getattr(self, name)
        elif name in self._params:
            return self._params[name].value
        else:
            raise ValueError(f"No parameter with name {name}")

    def getParams(self) -> Dict[str, any]:
        """Creates a dictionary with all parameters that can be set"""
        return {name: self.getParam(name) for name in self.fields}

    def __getattr__(self, name: str):
        # allow params to be accessed in the "classic" way
        if "_params" in self.__dict__ and name in self._params:
            return self._params[name].value
        else:
            raise AttributeError()

    def setParam(self, name: str, value: Any) -> None:
        """
        If there is a parameter with the given name update its value using the
        provided one, else ignore it.
        """
        if not name in self._fields:
            return
        if name in self._extra:
            setattr(self, name, value)
        elif name in self._params:
            self._params[name].value = value

    def setParams(self, **kwargs) -> None:
        """
        Sets the given parameters in the local configuration.
        Ignores parameters it does not contain.
        """
        for name, value in kwargs.items():
            self.setParam(name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        # allow params to be set using "classic" way, too
        if "_params" in self.__dict__ and name in self._params:
            self._params[name].value = value
        return super().__setattr__(name, value)

    def __repr__(self) -> str:
        return (
            self.name
            + "\n"
            + "\n".join(f"{param} : {self.getParam(param)}" for param in self.fields)
        )

    def _finishParams(self, i: int) -> None:
        """
        Function called during calls to update().
        Can be overwritten by subclasses to calculate private parameters.
        """
        pass

    def _bindParams(self, program: Program, i: int) -> None:
        """Binds the i-th configuration to the given program"""
        program.bindParams(**self._device[i])

    def update(self, i: int) -> None:
        """
        Updates the i-th configuration stored on the device using the pipeline's
        current state.
        """
        self._finishParams(i)
        for name, struct in self._local.items():
            memmove(self._device[i][name].memory, addressof(struct), sizeof(struct))

    @abstractmethod
    def run(self, i: int) -> List[Command]:
        """
        Creates a list of commands responsible for running the pipeline stage
        using the i-th configuration
        """
        pass


def runPipelineStage(stage: PipelineStage, i: int = 0, *, update: bool = True) -> None:
    """
    Runs the i-th configuration of the given pipeline stage and waits for it to
    finish. If update is True, calls `stage.update(i)` before running it.
    """
    if update:
        stage.update(i)
    beginSequence().AndList(stage.run(i)).Submit().wait()


class SourceCodeMixin(PipelineStage):
    """
    Base class for mixin that inject source code into other stages or programs
    while providing an interface to configure the code in the same way as
    regular pipeline stages.

    Parameters
    ----------
    params: {str: Structure}, default={}
        Dictionary of named ctype structure containing the stage's parameters.
        Each structure will be allocated on the CPU side and twice on the GPU
        for buffering. The latter can be bound in programs
    extra: {str}, default={}
        Set of extra parameter name, that can be set and retrieved using the
        stage api. Take precedence over parameters defined by structs. Should
        be be implemented in subclasses as properties.

    See Also
    --------
    PipelineStage
    """

    def __init__(
        self, params: Dict[str, type[Structure]] = {}, extra: Set[str] = set()
    ) -> None:
        super().__init__(params, extra)

    @property
    @abstractmethod
    def sourceCode(self) -> str:
        """Source code this mixin manages"""
        pass

    def bindParams(self, program: Program, i: int) -> None:
        """
        Binds params used by this mixin to the given program using the i-th
        configuration.
        """
        self._bindParams(program, i)

    def run(self, i: int) -> List[Command]:
        # Most mixin won't add any logic besides the source code
        # so an empty command list is a reasonable default
        return []


class RetrieveTensorStage(PipelineStage):
    """
    Utility stage retrieving a tensor to a local buffer.
    Keeps independent copies for each configuration.
    """

    name = "retrieve"

    def __init__(self, src: Tensor) -> None:
        super().__init__({})
        self._src = src
        # create local buffers
        self._buffers = [RawBuffer(src.size_bytes) for _ in range(2)]
        # create byte array to mimic std::span<std::byte>
        span = c_uint8 * src.size_bytes
        self._data = [span.from_address(buf.address) for buf in self._buffers]

    @property
    def src(self) -> Tensor:
        """Source tensor"""
        return self._src

    def address(self, i: int) -> int:
        """Returns the memory address of the i-th configuration"""
        return self._buffers[i].address

    def buffer(self, i: int) -> RawBuffer:
        """Returns the i-th buffer"""
        return self._buffers[i]

    def view(self, i: int, dtype: DTypeLike = float32) -> NDArray:
        """Interprets the retrieved tensor as array of given type"""
        return frombuffer(self._data[i], dtype)

    def run(self, i: int) -> List[Command]:
        return [retrieveTensor(self._src, self._buffers[i])]


class UpdateTensorStage(PipelineStage):
    """
    Utility stage updating a tensor from a local buffer. Keeps an independent
    source buffer for each configuration to allow concurrent pipeline reading
    and buffer writing.

    To make the buffer update part of the pipeline, consider overwriting
    `_finishParams(config)` in a subclass.
    """

    name = "update"

    def __init__(self, dst: Tensor) -> None:
        super().__init__({})
        self._dst = dst
        # create local buffers
        self._buffers = [RawBuffer(dst.size_bytes) for _ in range(2)]
        # create byte array to mimic std::span<std::byte>
        span = c_uint8 * dst.size_bytes
        self._data = [span.from_address(buf.address) for buf in self._buffers]

    @property
    def dst(self) -> Tensor:
        """Target tensor"""
        return self._dst

    def address(self, i: int) -> int:
        """Returns the memory address of the i-th configuration"""
        return self._buffers[i].address

    def buffer(self, i: int) -> RawBuffer:
        """Returns the i-th buffer"""
        return self._buffers[i]

    def view(self, i: int, dtype: DTypeLike = float32) -> NDArray:
        """Interprets the retrieved tensor as array of given type"""
        return frombuffer(self._data[i], dtype)

    def run(self, i: int) -> List[Command]:
        return [updateTensor(self._buffers[i], self._dst)]


class Pipeline:
    """
    Pipelines contain a sequence of named pipeline stages, manages their states
    as well as providing subroutines to be used for scheduling tasks to be fed
    into the pipeline.
    If no name is provided, the default name of the stage is used. If this name
    is already in use, a increasing number gets added, i.e. the name `stage`
    becomes `stage`, `stage2`, `stage3`, etc.

    Changes to the stages can be issued via a dict assigning a parameter path
    "{stage_name}__{parameter]" to change the parameter in a specific stage or
    just "parameter" to apply the change to all stages with that property.

    Parameters
    ----------
    stages: (PipelineStage | (name, PipelineStage))[]
        sequence of stages the pipeline consists of. Each element can optionally
        specify a name used for updating properties. If no name is provided
        (i.e. not a tuple), it get the name "stage{i}" where i is the stage's
        position in the pipeline
    """

    def __init__(
        self, stages: List[Union[PipelineStage, Tuple[str, PipelineStage]]]
    ) -> None:
        # create stage list and dict
        stages = [s if isinstance(s, tuple) else (s.name, s) for s in stages]
        self._stageList = []  # for ordering
        self._stageDict = {}  # for lookup
        name_counter = {}
        for name, stage in stages:
            # get unique name
            if name in name_counter:
                name_counter[name] += 1
                name = f"{stage.name}{name_counter}"
            else:
                name_counter[name] = 1
            # put stage in list and dict
            self._stageList.append((name, stage))
            self._stageDict[name] = stage

        # create subroutines
        self._subroutines = [
            createSubroutine(
                list(chain.from_iterable(stage.run(i) for _, stage in self._stageList)),
                simultaneous=True,
            )
            for i in range(2)
        ]

    @property
    def stages(self) -> List[Tuple[str, PipelineStage]]:
        """Sequence of named pipeline stages."""
        # create copy to be safe
        return list(self._stageList)

    def getSubroutine(self, i: int) -> Subroutine:
        """
        Returns the subroutine responsible for running the pipeline using
        the i-th configuration
        """
        return self._subroutines[i]

    def getParams(self) -> Dict[str, Any]:
        """
        Collect the parameters of all stages and returns their names and values
        stored in a dict.
        """
        return {
            stageName + "__" + paramName: value
            for stageName, stage in self._stageList
            for paramName, value in stage.getParams().items()
        }

    def setParams(self, **params) -> None:
        """
        Sets the parameters in the stages. A specific stage can be selected via
        `{name}__{parameter}`. If instead only the parameter name is provided,
        the parameter is applied to all stages.
        """
        for name, value in params.items():
            if "__" in name:
                stage, param = name.split("__", 1)
                if not stage in self._stageDict:
                    warnings.warn(f'There is no stage "{stage}" in this pipeline!')
                self._stageDict[stage].setParam(param, value)
            else:
                for _, stage in self._stageList:
                    stage.setParam(name, value)

    def update(self, i: int) -> None:
        """
        Updates the i-th configuration of all stages using their current
        configuration. Note that this does not check if the configuration is
        currently in use and results in undefined behavior if so.
        """
        for _, stage in self._stageList:
            stage.update(i)

    def runAsync(self, i: int, *, update: bool = True) -> Submission:
        """
        Runs the pipeline using the i-th configuration and returns a
        `Submission` which can be used to wait for the pipeline to finish. If
        update is True, updates all stages using their current state before
        running the pipeline. Note that this does not check if the configuration
        is currently in use and results in undefined behavior if so.
        """
        if update:
            self.update(i)
        return beginSequence().And(self.getSubroutine(i)).Submit()

    def run(self, i: int, *, update: bool = True) -> None:
        """
        Runs the pipeline using the i-th configuration and waits for it to
        finish. If update is True, updates all stages using their current state
        before running the pipeline. Note that this does not check if the
        configuration is currently in use and results in undefined behavior if
        so.
        """
        self.runAsync(i, update=update).wait()


def runPipeline(
    stages: Iterable[PipelineStage], i: int = 0, *, update: bool = True
) -> None:
    """
    Runs the i-th configuration of the stages as if they make up a pipeline in
    given order and waits for it to finish. If update is True, calls `update(i)`
    on each stage before running them.

    Calling this function may be faster than creating a `Pipeline`, but is most
    likely slower if called repeatedly. Consider creating a `Pipeline` in that
    case.
    """
    if update:
        for stage in stages:
            stage.update(i)
    beginSequence().AndList(
        list(chain.from_iterable(stage.run(i) for stage in stages))
    ).Submit().wait()


class _CounterWorkerThread:
    """
    Util class for worker threads used by the scheduler. Runs a function as many
    times as specified, while allowing to increase this value. If it is reached
    it suspends the thread ready to run more loops.
    """

    def __init__(self, fn: Callable[[int], None]) -> None:
        self._thread = Thread(target=self._loop, daemon=True)
        self._isSuspending = True
        self._suspendLock = Lock()
        self._wakeEvent = Event()
        self._counter = 0
        self._target = 0
        self._fn = fn
        # start thread
        self._thread.start()

    @property
    def count(self) -> int:
        """Number of finished iterations"""
        return self._counter

    @property
    def total(self) -> int:
        """Total number of issued iterations"""
        return self._target

    def run(self, n: int) -> None:
        """Increases the loop target"""
        self._target += n
        # check if we need to wake
        with self._suspendLock:
            if self._isSuspending:
                self._wakeEvent.set()

    def _loop(self) -> None:
        """Thread body"""
        # infinite loop -> must run inside a daemon thread
        while True:
            # suspend if necessary
            if self._isSuspending:
                self._wakeEvent.wait()
                # resume work
                self._wakeEvent.clear()
                self._isSuspending = False
            # quick and unsafe check
            if self._counter >= self._target:
                # might want to suspend -> check again, but safe
                with self._suspendLock:
                    if self._counter >= self._target:
                        self._isSuspending = True
                        continue
            # run function
            self._fn(self._counter)
            self._counter += 1


class PipelineScheduler:
    """
    Schedules tasks into a pipeline and orchestrates the processing of the
    pipeline results. Bundles tasks into a single batch submission making it
    more efficient than repeatedly calling `run` or `runAsync` on the pipeline.

    Scheduling tasks happens completely in the background using multithreading
    allowing the calling code to do other work, e.g. preparing the next
    submission to be issued to the scheduler, but also provides function to wait
    on the completion of a certain or all tasks.

    New tasks can be issued while previous ones are still processed.

    The scheduler requires exclusive access to two configurations, but may
    otherwise share the pipeline.

    Parameters
    ----------
    pipeline: Pipeline
        Pipeline onto which to schedule tasks
    queueSize: int, default=0
        Size of the task queue. Size of 0 means infinite.
    processFn: Callable( (config: int, task: int) -> None ), default=None
        Function to be called after each finished task on the pipeline with
        config being the pipeline's config used and task the increasing task
        count processed, i.e. 0 for the first task, 1 for the second and so on.
        The scheduler ensures tasks using the same configuration to wait until
        processing finished. Will run in its own thread.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        queueSize: int = 0,
        processFn: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        # save params
        self._queueSize = queueSize
        self._totalTasks = 0
        self._pipeline = pipeline
        self._pipelineTimeline = Timeline()
        # create queue worker
        self._updateTimeline = Timeline()
        self._updateWorker = _CounterWorkerThread(self._update)
        self._updateQueue = Queue()
        # create process worker if needed
        self._processFn = processFn
        if processFn is None:
            self._processWorker = None
            self._processTimeline = None
        else:
            self._processWorker = _CounterWorkerThread(self._process)
            self._processTimeline = Timeline()

    @property
    def pipeline(self) -> Pipeline:
        """The underlying pipeline into which tasks are orchestrated"""
        return self._pipeline

    @property
    def queueSize(self) -> int:
        """
        Capacity of the internal queue used for scheduling tasks.
        A value of zero means infinite.
        """
        return self._queueSize

    @property
    def totalTasks(self) -> int:
        """Total number of tasks scheduled"""
        return self._totalTasks

    @property
    def tasksScheduled(self) -> int:
        """Approximate number of tasks scheduled"""
        return self._updateQueue.qsize()

    @property
    def tasksFinished(self) -> int:
        """Returns the number of finished tasks processed by the timeline"""
        if self._processWorker is None:
            return self._pipelineTimeline.value
        else:
            return self._processWorker.count

    def schedule(
        self, tasks: Iterable[Dict[str, Any]], *, timeout: Optional[float] = None
    ) -> Tuple[int, Submission]:
        """
        Schedules the given list of tasks to be processed by the pipeline after
        previous tasks submissions have finished.

        Parameters
        ----------
        tasks: Iterable[Dict[str, Any]]
            Sequence of tasks to schedule onto the pipeline defined by the set
            of parameters to apply to the stages.
        timeout: float | None, default=None
            Timeout in seconds for waiting on free space in the queue.
            If None, waits indefinitely.

        Returns
        -------
        nSubmitted: int
            Number of tasks actually submitted.
        submission: Submission | None
            The submission created for submitting work to the GPU. Can be used
            to wait on the tasks to finish or query the final task index via
            `submission.finalStep`.
            None if nSubmitted == 0.
        """
        # put task onto queue
        n = 0
        builder = None
        for task in tasks:
            # try to enlist in queue
            try:
                self._updateQueue.put(task, timeout=timeout)
            except:
                break

            # lazy create builder
            if builder is None:
                builder = beginSequence(self._pipelineTimeline, self._totalTasks)

            # issue wait on previous task
            builder.WaitFor(self._pipelineTimeline, self._totalTasks)
            # issue wait on update thread
            builder.WaitFor(self._updateTimeline, self._totalTasks + 1)
            # issue wait on process thread is present
            if self._processTimeline is not None and self._totalTasks >= 2:
                # double buffered -> wait for the processing of tasks scheduled
                # two earlier to finish to make sure we don't overwrite the
                # result it tries to process
                builder.WaitFor(self._processTimeline, self._totalTasks - 1)
            # run task
            i = self._totalTasks % 2
            builder.And(self._pipeline.getSubroutine(i))

            # update counters
            n += 1
            self._totalTasks += 1

        # enqueued anything?
        if n == 0:
            return (0, None)

        # submit work
        submission = builder.Submit()
        # just to ensure we're not breaking stuff in the future
        assert submission.forgettable

        # update worker threads
        self._updateWorker.run(n)
        if self._processWorker is not None:
            self._processWorker.run(n)

        # return result
        return (n, submission)

    def wait(self, task: Optional[int] = None) -> None:
        """
        Waits on the given task to finish. Blocks the calling code.
        If task is None, waits on the last scheduled task.

        Waiting on a task not scheduled may result in a deadlock if no further
        thread schedule tasks.
        """
        if task is None:
            task = self.totalTasks
        if self._processTimeline is None:
            self._pipelineTimeline.wait(task)
        else:
            self._processTimeline.wait(task)

    def waitTimeout(self, task: Optional[int] = None, *, timeout: int = 1000) -> bool:
        """
        Waits on the given task to finish for at most `timeout` nanoseconds.

        Parameters
        ----------
        task: int | None, default=None
            Position of the task to wait on. Waits on the last added one if None.
        timeout: int, default=1000
            Timeout in nanoseconds. The driver is free to round this value up to
            its internal resolution. May be zero to query the tasks state
            immediately.

        Returns
        -------
        finished: bool
            True, if the given task finished.
        """
        if task is None:
            task = self.totalTasks
        if self._processTimeline is None:
            self._pipelineTimeline.waitTimeout(task, timeout)
        else:
            self._processTimeline.waitTimeout(task, timeout)

    def _update(self, n: int) -> None:
        """Internal update thread body"""
        # fetch next task (don't need to block, as this is handled by the worker)
        task: Optional[Dict[str, Any]] = self._updateQueue.get(block=False)
        # update pipeline
        self._pipeline.setParams(**task)
        # wait for the i-th config to be safe to update
        if n >= 2:
            self._pipelineTimeline.wait(n - 1)
        # update config
        # eventually calls user provided functions
        # -> treat as evil to prevent from deadlocking timeline
        try:
            self._pipeline.update(n % 2)
        except Exception as ex:
            warnings.warn(f"Exception raised while preparing task {n}:\n{ex}")
        # advance timeline
        self._updateTimeline.value = n + 1

    def _process(self, n: int) -> None:
        """Internal process thread body"""
        # wait on task to finish
        self._pipelineTimeline.wait(n + 1)
        # process
        # external provided function
        # -> treat as evil to prevent from deadlocking timeline
        try:
            self._processFn(n % 2, n)
        except Exception as ex:
            warnings.warn(f"Exception raised while processing task {n}:\n{ex}")
        # advance timeline
        self._processTimeline.value = n + 1
