"""
distributed_rules.py
====================
Experta rules for distributed and multi-GPU training issues.

Rule IDs: DIST_001 - DIST_010
"""

from experta import KnowledgeEngine, Rule, NOT

from models.facts import (
    # Context
    UsesDistributedTraining,
    # Symptoms
    MultiGpuThroughputLow,
    GpuUnderutilization,
    BatchNormDesync,
    GradientSyncSlow,
    PerDeviceBatchTooSmallObserved,
    SlowConvergence,
    # Causes
    DataLoadingBottleneck,
    ImproperBatchNormSync,
    GradientSyncOverhead,
    LearningRateNotScaled,
    PerDeviceBatchTooSmall,
    # Recommendations
    IncreaseDataLoaderWorkers,
    UseSyncBatchNorm,
    UseGradientAccumulation,
    ScaleLearningRateByWorldSize,
    # XAI
    Explanation,
)


class DistributedRules(KnowledgeEngine):
    """Rule set covering distributed-training diagnostics."""

    # DIST_001 - Distributed + low throughput + GPU idle -> DataLoadingBottleneck
    @Rule(
        UsesDistributedTraining(),
        MultiGpuThroughputLow(),
        GpuUnderutilization(),
        NOT(DataLoadingBottleneck()),
    )
    def dist_001_data_loading_bottleneck(self) -> None:
        self.declare(DataLoadingBottleneck())
        self.declare(
            Explanation(
                rule_id="DIST_001",
                triggered_by="UsesDistributedTraining,MultiGpuThroughputLow,GpuUnderutilization",
                derived="DataLoadingBottleneck",
                confidence=0.87,
                explanation=(
                    "Low multi-GPU throughput with idle devices usually means "
                    "the input pipeline cannot feed workers quickly enough."
                ),
            )
        )

    # DIST_002 - Distributed + BatchNormDesync -> ImproperBatchNormSync
    @Rule(
        UsesDistributedTraining(),
        BatchNormDesync(),
        NOT(ImproperBatchNormSync()),
    )
    def dist_002_batch_norm_sync(self) -> None:
        self.declare(ImproperBatchNormSync())
        self.declare(
            Explanation(
                rule_id="DIST_002",
                triggered_by="UsesDistributedTraining,BatchNormDesync",
                derived="ImproperBatchNormSync",
                confidence=0.9,
                explanation=(
                    "Desynchronized BatchNorm statistics across workers point "
                    "directly to missing or incorrect synchronized BatchNorm."
                ),
            )
        )

    # DIST_003 - Distributed + GradientSyncSlow -> GradientSyncOverhead
    @Rule(
        UsesDistributedTraining(),
        GradientSyncSlow(),
        NOT(GradientSyncOverhead()),
    )
    def dist_003_gradient_sync_overhead(self) -> None:
        self.declare(GradientSyncOverhead())
        self.declare(
            Explanation(
                rule_id="DIST_003",
                triggered_by="UsesDistributedTraining,GradientSyncSlow",
                derived="GradientSyncOverhead",
                confidence=0.86,
                explanation=(
                    "If gradient synchronization dominates step time, the "
                    "distributed setup is paying too much communication overhead."
                ),
            )
        )

    # DIST_004 - Distributed + too-small per-device batch -> PerDeviceBatchTooSmall
    @Rule(
        UsesDistributedTraining(),
        PerDeviceBatchTooSmallObserved(),
        NOT(PerDeviceBatchTooSmall()),
    )
    def dist_004_per_device_batch(self) -> None:
        self.declare(PerDeviceBatchTooSmall())
        self.declare(
            Explanation(
                rule_id="DIST_004",
                triggered_by="UsesDistributedTraining,PerDeviceBatchTooSmallObserved",
                derived="PerDeviceBatchTooSmall",
                confidence=0.84,
                explanation=(
                    "Very small per-device batches produce noisy gradients and "
                    "unstable normalization statistics in distributed training."
                ),
            )
        )

    # DIST_005 - Distributed + SlowConvergence -> LearningRateNotScaled
    @Rule(
        UsesDistributedTraining(),
        SlowConvergence(),
        NOT(LearningRateNotScaled()),
    )
    def dist_005_lr_not_scaled(self) -> None:
        self.declare(LearningRateNotScaled())
        self.declare(
            Explanation(
                rule_id="DIST_005",
                triggered_by="UsesDistributedTraining,SlowConvergence",
                derived="LearningRateNotScaled",
                confidence=0.68,
                explanation=(
                    "Slow convergence after increasing worker count can mean "
                    "the learning rate was not adjusted for the larger global batch."
                ),
            )
        )

    # DIST_006 - Distributed + DataLoadingBottleneck -> IncreaseDataLoaderWorkers
    @Rule(
        UsesDistributedTraining(),
        DataLoadingBottleneck(),
        NOT(IncreaseDataLoaderWorkers()),
    )
    def dist_006_increase_workers(self) -> None:
        self.declare(IncreaseDataLoaderWorkers())
        self.declare(
            Explanation(
                rule_id="DIST_006",
                triggered_by="UsesDistributedTraining,DataLoadingBottleneck",
                derived="IncreaseDataLoaderWorkers",
                confidence=0.84,
                explanation=(
                    "Increasing loader workers, prefetching, or pinned-memory "
                    "transfer can keep distributed workers fed with batches."
                ),
            )
        )

    # DIST_007 - Distributed + ImproperBatchNormSync -> UseSyncBatchNorm
    @Rule(
        UsesDistributedTraining(),
        ImproperBatchNormSync(),
        NOT(UseSyncBatchNorm()),
    )
    def dist_007_sync_batch_norm(self) -> None:
        self.declare(UseSyncBatchNorm())
        self.declare(
            Explanation(
                rule_id="DIST_007",
                triggered_by="UsesDistributedTraining,ImproperBatchNormSync",
                derived="UseSyncBatchNorm",
                confidence=0.9,
                explanation=(
                    "Synchronized BatchNorm keeps normalization statistics "
                    "consistent across workers and removes per-replica drift."
                ),
            )
        )

    # DIST_008 - Distributed + GradientSyncOverhead -> UseGradientAccumulation
    @Rule(
        UsesDistributedTraining(),
        GradientSyncOverhead(),
        NOT(UseGradientAccumulation()),
    )
    def dist_008_gradient_accumulation(self) -> None:
        self.declare(UseGradientAccumulation())
        self.declare(
            Explanation(
                rule_id="DIST_008",
                triggered_by="UsesDistributedTraining,GradientSyncOverhead",
                derived="UseGradientAccumulation",
                confidence=0.8,
                explanation=(
                    "Accumulating gradients over microbatches reduces how often "
                    "workers synchronize, improving communication efficiency."
                ),
            )
        )

    # DIST_009 - Distributed + LearningRateNotScaled -> ScaleLearningRateByWorldSize
    @Rule(
        UsesDistributedTraining(),
        LearningRateNotScaled(),
        NOT(ScaleLearningRateByWorldSize()),
    )
    def dist_009_scale_lr(self) -> None:
        self.declare(ScaleLearningRateByWorldSize())
        self.declare(
            Explanation(
                rule_id="DIST_009",
                triggered_by="UsesDistributedTraining,LearningRateNotScaled",
                derived="ScaleLearningRateByWorldSize",
                confidence=0.78,
                explanation=(
                    "Scaling the learning rate with distributed world size "
                    "keeps optimizer step sizes aligned with the global batch."
                ),
            )
        )

    # DIST_010 - Distributed + PerDeviceBatchTooSmall -> UseGradientAccumulation
    @Rule(
        UsesDistributedTraining(),
        PerDeviceBatchTooSmall(),
        NOT(UseGradientAccumulation()),
    )
    def dist_010_accumulate_small_batch(self) -> None:
        self.declare(UseGradientAccumulation())
        self.declare(
            Explanation(
                rule_id="DIST_010",
                triggered_by="UsesDistributedTraining,PerDeviceBatchTooSmall",
                derived="UseGradientAccumulation",
                confidence=0.82,
                explanation=(
                    "Gradient accumulation increases the effective batch size "
                    "when each device cannot hold enough samples at once."
                ),
            )
        )
