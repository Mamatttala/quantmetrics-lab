"""Module D — regime detection over HTTP.

The request is one object rather than a set of query parameters because every field is
a modelling decision that has to be recorded together in the output's lineage. A regime
result is only interpretable next to the thresholds and window lengths that produced it;
storing the frame without them would make the output unreproducible, which is the one
thing a research artifact must never be.

The two confidence thresholds are separate fields, not one. The specification gives both
a 55% rule ("model Transition when no state holds a clear majority") and a 60% rule ("a
switch requires at least this much confidence"), and they constrain different things:
the first decides whether the *current* state is identifiable at all, the second decides
whether a *change* of state is permitted. Collapsing them into one knob would silently
pick one of the two behaviours and make the other unreachable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from app.models.enums import JobStatus, RegimeFeature, RegimeModel


class RegimeDetectRequest(BaseModel):
    """Parameters for one detection run.

    Defaults follow the specification except ``n_states``, which defaults to 4 rather
    than 3: the output schema names four states, and a 3-state fit cannot populate four
    columns without inventing a mapping for the missing one. See DECISIONS §73.
    """

    dataset_id: str

    model_type: RegimeModel = RegimeModel.HMM
    n_states: int = Field(default=4, ge=2, le=5, description="Latent states to fit.")

    #: Which observations the model is fitted on. Order is preserved in the report so a
    #: result can be read back against the request that produced it.
    features: list[RegimeFeature] = Field(
        default_factory=lambda: [
            RegimeFeature.LOG_RETURN,
            RegimeFeature.REALIZED_VOLATILITY,
        ]
    )

    #: Anti-thrashing. A state is held until it has persisted this many bars AND the
    #: candidate replacement is confident enough.
    min_state_duration: int = Field(default=5, ge=1, le=1000)
    switching_cost_penalty: float = Field(
        default=0.005,
        ge=0.0,
        le=1.0,
        description="Cost charged per position change when scoring Net Value-Add.",
    )

    #: Annualised realized volatility above which the rules-based override declares
    #: crisis regardless of what the statistical model says.
    crisis_threshold: float = Field(default=0.35, gt=0.0, le=10.0)
    override_probability: float = Field(
        default=0.90,
        gt=0.0,
        le=1.0,
        description="Crisis probability written in when the override fires.",
    )

    #: Below this, no state is identifiable and the row is labelled Transition.
    transition_confidence: float = Field(default=0.55, gt=0.0, lt=1.0)
    #: A change of committed state requires at least this much confidence.
    commit_confidence: float = Field(default=0.60, gt=0.0, lt=1.0)

    #: Walk-forward schedule. ``refit_every`` is measured in **bars from the start of the
    #: series**, never as a fraction of its length: anchoring the schedule to the series
    #: length would make the first 80% of a run depend on what the last 20% contains
    #: (it would move every refit point), which is exactly the property the causality
    #: test exists to catch. See DECISIONS §74.
    min_train_bars: int = Field(default=250, ge=20)
    refit_every: int = Field(default=250, ge=1)

    #: Trailing window, in bars, for the realized-volatility observation and for the
    #: crisis override. The override reads the same series the model does, so the two
    #: cannot disagree about how volatile the market currently is.
    volatility_window: int = Field(default=24, ge=2)

    #: When false the run is computed and reported but nothing is written — the same
    #: preview contract the cleaning and merge endpoints use.
    write: bool = True
    name: str | None = None

    @model_validator(mode="after")
    def _check(self) -> RegimeDetectRequest:
        if not self.features:
            raise ValueError("Select at least one feature to fit on.")
        if len(set(self.features)) != len(self.features):
            raise ValueError("A feature is selected more than once.")
        if self.commit_confidence < self.transition_confidence:
            # Otherwise a state could clear the "is it identifiable" bar and still not
            # be permitted to become the committed label, leaving a row that is neither
            # ambiguous nor assignable.
            raise ValueError(
                "commit_confidence must be at least transition_confidence; "
                f"got {self.commit_confidence} < {self.transition_confidence}."
            )
        if self.n_states < 2:
            raise ValueError("A regime model needs at least two states.")
        return self


class RegimeRunAccepted(BaseModel):
    """What ``POST /api/regime/detect`` returns immediately.

    The work is not done when this is returned. A caller that treats it as a result will
    read an empty frame, so the status field is mandatory rather than defaulted.
    """

    job_id: str
    status: JobStatus
    poll: str
