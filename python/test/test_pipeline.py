import hephaistos as hp
import hephaistos.pipeline as pl
import numpy as np

from ctypes import Structure, c_int32
from os.path import dirname, join

from typing import List


def test_runPipelineStage():
    # create tensor
    tensor = hp.IntTensor(64)
    hp.execute(hp.clearTensor(tensor, data=42))

    # create retrieve stage
    retriever = pl.RetrieveTensorStage(tensor)
    pl.runPipelineStage(retriever, 0)

    # create different value for second buffer
    hp.execute(hp.clearTensor(tensor, data=23, size=32 * 4))
    pl.runPipelineStage(retriever, 1)

    # check result
    expected = np.zeros(64) + 42
    assert np.all(retriever.view(0, np.int32) == expected)
    expected[:32] = 23
    assert np.all(retriever.view(1, np.int32) == expected)


def test_runPipeline():
    # create tensor
    tensor = hp.IntTensor(64)
    # create pipeline
    updater = pl.UpdateTensorStage(tensor)
    retriever = pl.RetrieveTensorStage(tensor)

    # fill buffers
    rng = np.random.default_rng(0xC0FFEE)
    b1 = rng.integers(-100, 100, 64, np.int32)
    b2 = rng.integers(-100, 100, 64, np.int32)
    # copy buffers
    updater.view(0, np.int32)[:] = b1
    updater.view(1, np.int32)[:] = b2

    # run pipeline stage
    pl.runPipeline([updater, retriever], 0)
    pl.runPipeline([updater, retriever], 1)

    # check result
    assert np.all(retriever.view(0, np.int32) == b1)
    assert np.all(retriever.view(1, np.int32) == b2)


class PipelineTestStage(pl.PipelineStage):
    class Params(Structure):
        _fields_ = [("m", c_int32), ("b", c_int32), ("_dummy", c_int32)]

    name = "test"

    def __init__(self) -> None:
        super().__init__({"Params": self.Params})
        # load shader
        code = None
        shader_path = join(dirname(__file__), "shader/pipeline_test.spv")
        with open(shader_path, "rb") as f:
            code = f.read()
        # create program
        self._program = hp.Program(code)
        # create tensor
        self.tensor = hp.IntTensor(256)
        self._program.bindParams(Output=self.tensor)

    def run(self, i: int) -> List:
        self._bindParams(self._program, i)
        return [self._program.dispatch(256 // 32)]


def test_pipeline():
    # create pipeline
    comp = PipelineTestStage()
    retr = pl.RetrieveTensorStage(comp.tensor)
    pipeline = pl.Pipeline([comp, retr])

    # set params
    params = {"m": 2, "test__b": 15}
    pipeline.setParams(**params)
    # check params
    assert pipeline.getParams() == {"test__m": 2, "test__b": 15}

    # run pipeline
    pipeline.run(1)

    # check result
    expected = np.arange(256) * 2.0 + 15.0
    assert np.all(retr.view(1, np.int32) == expected)


def test_scheduler_single():
    # create pipeline
    comp = PipelineTestStage()
    retr = pl.RetrieveTensorStage(comp.tensor)
    pipeline = pl.Pipeline([comp, retr])

    # create processing function
    results = []
    process = lambda config, batch, args: results.append(
        retr.view(config, np.int32).copy()
    )

    # create scheduler
    scheduler = pl.PipelineScheduler(pipeline, processFn=process)

    # create two list of tasks
    m1 = [2 * x + 1 for x in range(5)]
    b1 = [50 * (x + 1) for x in range(5)]
    m2 = [-5 * x - 1 for x in range(5)]
    b2 = [100 * (x + 1) for x in range(5)]
    # create two task list
    t1 = [{"test__m": m1[i], "b": b1[i]} for i in range(len(m1))]
    t2 = [{"test__m": m2[i], "b": b2[i]} for i in range(len(m2))]
    N = len(t1) + len(t2)

    # submit both tasks
    scheduler.schedule(t1)
    scheduler.schedule(t2)
    # check if task scheduler
    assert scheduler.totalTasks == N

    # wait on scheduler to finish
    scheduler.wait()

    # check if everything was processed
    assert len(results) == N
    assert scheduler.tasksFinished == N

    # check results
    for i in range(len(m1)):
        m, b = m1[i], b1[i]
        expected = np.arange(256) * m + b
        assert np.all(results[i] == expected)
    for i in range(len(m2)):
        m, b = m2[i], b2[i]
        expected = np.arange(256) * m + b
        assert np.all(results[i + len(m1)] == expected)

    # destroy scheduler
    scheduler.destroy()
    assert scheduler.destroyed


def test_scheduler_multi():
    # create multiple pipelines
    t = PipelineTestStage()
    p1 = pl.Pipeline([t, pl.RetrieveTensorStage(t.tensor)])
    p2 = pl.Pipeline([t, pl.RetrieveTensorStage(t.tensor)])
    p3 = pl.Pipeline([t, pl.RetrieveTensorStage(t.tensor)])

    # create scheduler
    scheduler = pl.PipelineScheduler({"p1": p1, "p2": p2, "p3": p3})

    # create tasks
    N = 8
    m = [2 * x + 1 for x in range(N)]
    b = [50 * (x + 1) for x in range(N)]
    tasks = [(f"p{i % 3 + 1}", {"test__m": m[i], "b": b[i]}) for i in range(N)]
    # schedule tasks
    scheduler.schedule(tasks)
    # check if everything was scheduled
    assert scheduler.totalTasks == N

    # wait on scheduler to finish
    scheduler.wait()

    # check if everything was processed
    assert scheduler.tasksFinished == N

    # fetch results
    r1 = p1.stages[1][1].view(1, np.int32)
    r2 = p2.stages[1][1].view(0, np.int32)
    r3 = p3.stages[1][1].view(1, np.int32)
    # check results
    x = np.arange(256)
    assert np.all(r1 == m[3] * x + b[3])
    assert np.all(r2 == m[4] * x + b[4])
    assert np.all(r3 == m[5] * x + b[5])

    # destroy scheduler
    scheduler.destroy()
    assert scheduler.destroyed


def test_dynamicTask():
    # create pipeline
    comp = PipelineTestStage()
    retr = pl.RetrieveTensorStage(comp.tensor)
    pipeline = pl.Pipeline([comp, retr])
    params = {"m": 5, "b": 12}
    expected = np.arange(256) * params["m"] + params["b"]

    # create dynamic task
    class DummyDynamicTask(pl.DynamicTask):
        def __init__(self) -> None:
            super().__init__(params, initialBatchCount=4)
            self.counter = 0
            self.failed = False

        def processBatch(self, config: int) -> int:
            self.counter += 1
            # check result to see if batch was issued correctly
            result = retr.view(config, np.int32)
            if not np.all(result == expected):
                self.failed = True

            # issue new batches
            if self.counter == 1:
                return 2
            elif self.counter == 4:
                return 3
            else:
                return 0

    # create scheduler + issue task
    tasks = [
        DummyDynamicTask(),
        DummyDynamicTask(),
        DummyDynamicTask(),
    ]
    scheduler = pl.DynamicTaskScheduler(pipeline)
    scheduler.scheduleTasks(tasks)

    # call both wait functions
    scheduler.waitNext()
    assert tasks[0].isFinished
    scheduler.waitAll()

    # check task
    assert all(not t.failed for t in tasks)
    assert all(t.counter == 9 for t in tasks)
