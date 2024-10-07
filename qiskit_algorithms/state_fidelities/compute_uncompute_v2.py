# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Compute-uncompute fidelity interface using primitives V2
"""

from __future__ import annotations
from collections.abc import Sequence
from copy import copy
import time

from qiskit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2, PrimitiveResult
from qiskit.providers import Options
from qiskit.result import QuasiDistribution
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime.ibm_backend import IBMBackend
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.runtime_job_v2 import RuntimeJobV2

from ..exceptions import AlgorithmError
from .base_state_fidelity import BaseStateFidelity
from .state_fidelity_result import StateFidelityResult
from ..algorithm_job import AlgorithmJob


class ComputeUncomputeV2(BaseStateFidelity):
    r"""
    This class leverages the sampler primitive to calculate the state
    fidelity of two quantum circuits following the compute-uncompute
    method (see [1] for further reference).
    The fidelity can be defined as the state overlap.

    .. math::

            |\langle\psi(x)|\phi(y)\rangle|^2

    where :math:`x` and :math:`y` are optional parametrizations of the
    states :math:`\psi` and :math:`\phi` prepared by the circuits
    ``circuit_1`` and ``circuit_2``, respectively.

    **Reference:**
    [1] Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala,
    A., Chow, J. M., & Gambetta, J. M. (2019). Supervised learning
    with quantum-enhanced feature spaces. Nature, 567(7747), 209-212.
    `arXiv:1804.11326v2 [quant-ph] <https://arxiv.org/pdf/1804.11326.pdf>`_

    """

    def __init__(
        self,
        sampler: BaseSamplerV2,
        service: QiskitRuntimeService | None = None,
        backend: IBMBackend | None = None,
        job_id: str | None = None,
    ) -> None:
        r"""
        Args:
            sampler: Sampler primitive instance V2.
            service: Qiskit Runtime Service to identify QPU access.
            backend: QPU backend to run quantum circuit.
            job_id: Calling result from finished job. Default is None to run
                the job in QPU.

        Raises:
            ValueError: If the sampler is not an instance of ``BaseSamplerV2``.
        """
        options = None
        local = None
        if not isinstance(sampler, BaseSamplerV2):
            raise ValueError(
                f"The sampler should be an instance of BaseSamplerV2, " f"but got {type(sampler)}"
            )
        self._sampler: BaseSamplerV2 = sampler
        self._service = service
        self._backend = backend
        self._job_id = job_id
        self._local = local
        self._default_options = Options()
        if options is not None:
            self._default_options.update_options(**options)
        super().__init__()

    def create_fidelity_circuit(
        self, circuit_1: QuantumCircuit, circuit_2: QuantumCircuit
    ) -> QuantumCircuit:
        """
        Combines ``circuit_1`` and ``circuit_2`` to create the
        fidelity circuit following the compute-uncompute method.

        Args:
            circuit_1: (Parametrized) quantum circuit.
            circuit_2: (Parametrized) quantum circuit.

        Returns:
            The fidelity quantum circuit corresponding to circuit_1 and circuit_2.
        """
        if len(circuit_1.clbits) > 0:
            circuit_1.remove_final_measurements()
        if len(circuit_2.clbits) > 0:
            circuit_2.remove_final_measurements()

        circuit = circuit_1.compose(circuit_2.inverse())
        circuit.measure_all()
        return circuit

    def _run(
        self,
        circuits_1: QuantumCircuit | Sequence[QuantumCircuit],
        circuits_2: QuantumCircuit | Sequence[QuantumCircuit],
        values_1: Sequence[float] | Sequence[Sequence[float]] | None = None,
        values_2: Sequence[float] | Sequence[Sequence[float]] | None = None,
        **options,
    ) -> AlgorithmJob:
        r"""
        Computes the state overlap (fidelity) calculation between two
        (parametrized) circuits (first and second) for a specific set of parameter
        values (first and second) following the compute-uncompute method.

        Args:
            circuits_1: (Parametrized) quantum circuits preparing :math:`|\psi\rangle`.
            circuits_2: (Parametrized) quantum circuits preparing :math:`|\phi\rangle`.
            values_1: Numerical parameters to be bound to the first circuits.
            values_2: Numerical parameters to be bound to the second circuits.
            options: Primitive backend runtime options used for circuit execution.
                    The order of priority is: options in ``run`` method > fidelity's
                    default options > primitive's default setting.
                    Higher priority setting overrides lower priority setting.

        Returns:
            An AlgorithmJob for the fidelity calculation.

        Raises:
            ValueError: At least one pair of circuits must be defined.
            AlgorithmError: If the sampler job is not completed successfully.
        """

        circuits = self._construct_circuits(circuits_1, circuits_2)
        if len(circuits) == 0:
            raise ValueError(
                "At least one pair of circuits must be defined to calculate the state overlap."
            )
        values = self._construct_value_list(circuits_1, circuits_2, values_1, values_2)

        # The priority of run options is as follows:
        # options in `evaluate` method > fidelity's default options >
        # primitive's default options.
        opts = copy(self._default_options)
        opts.update_options(**options)

        if self._job_id is None:
            pm = generate_preset_pass_manager(optimization_level=1, backend=self._backend)
            isa_circuits = []
            for circuit in circuits:
                isa_circuits.append(pm.run(circuit))
            sampler_job = self._sampler.run([list(zip(isa_circuits, values))], **opts.__dict__)
            self._job_id = sampler_job.job_id()
            while True:
                job_status = sampler_job.status()
                if job_status == "DONE":
                    print("QPU job completed successfully!")
                    break
                print("\rWaiting for QPU job to complete...", end="", flush=True)
                time.sleep(1)
            print("\r", end="", flush=True)
        else:
            sampler_job = self._service.job(self._job_id)
        local_opts = self._get_local_options(opts.__dict__)
        return AlgorithmJob(
            ComputeUncomputeV2._call, sampler_job, circuits, self._local, local_opts
        )

    @staticmethod
    def _call(
        job: RuntimeJobV2, circuits: Sequence[QuantumCircuit], local: bool, local_opts: Options
    ) -> PrimitiveResult:
        try:
            result = job.result()
        except Exception as exc:
            raise AlgorithmError("Sampler job failed!") from exc

        probability_distributions = []
        for pub_result in result:
            counts = pub_result.data.meas.get_counts()
            n = len(list(counts.keys())[0])
            new_count = {int(format(i, "012b"), 2): 0 for i in range(2**n)}
            new_count.update({int(k, 2): v for k, v in counts.items()})
            total_shots = sum(counts.values())
            probabilities = QuasiDistribution(
                {bitstring: count / total_shots for bitstring, count in counts.items()}
            )
            probability_distributions.append(probabilities)

        if local:
            raw_fidelities = [
                ComputeUncomputeV2._get_local_fidelity(prob_dist, circuit.num_qubits)
                for prob_dist, circuit in zip(probability_distributions, circuits)
            ]
        else:
            raw_fidelities = [
                ComputeUncomputeV2._get_global_fidelity(prob_dist)
                for prob_dist in probability_distributions
            ]
        fidelities = ComputeUncomputeV2._truncate_fidelities(raw_fidelities)

        return StateFidelityResult(
            fidelities=fidelities,
            raw_fidelities=raw_fidelities,
            metadata=result.metadata,
            options=local_opts,
        )

    @property
    def options(self) -> Options:
        """Return the union of estimator options setting and fidelity default options,
        where, if the same field is set in both, the fidelity's default options override
        the primitive's default setting.

        Returns:
            The fidelity default + estimator options.
        """
        return self._get_local_options(self._default_options.__dict__)

    def update_default_options(self, **options):
        """Update the fidelity's default options setting.

        Args:
            **options: The fields to update the default options.
        """

        self._default_options.update_options(**options)

    def _get_local_options(self, options: Options) -> Options:
        """Return the union of the primitive's default setting,
        the fidelity default options, and the options in the ``run`` method.
        The order of priority is: options in ``run`` method > fidelity's
                default options > primitive's default setting.

        Args:
            options: The fields to update the options

        Returns:
            The fidelity default + estimator + run options.
        """
        opts = copy(self._sampler.options)
        opts.update_options(**options)
        return opts

    @staticmethod
    def _get_global_fidelity(probability_distribution: dict[int, float]) -> float:
        """Process the probability distribution of a measurement to determine the
        global fidelity.

        Args:
            probability_distribution: Obtained from the measurement result

        Returns:
            The global fidelity.
        """
        return probability_distribution.get(0, 0)

    @staticmethod
    def _get_local_fidelity(probability_distribution: dict[int, float], num_qubits: int) -> float:
        """Process the probability distribution of a measurement to determine the
        local fidelity by averaging over single-qubit projectors.

        Args:
            probability_distribution: Obtained from the measurement result

        Returns:
            The local fidelity.
        """
        fidelity = 0.0
        for qubit in range(num_qubits):
            for bitstring, prob in probability_distribution.items():
                # Check whether the bit representing the current qubit is 0
                if not bitstring >> qubit & 1:
                    fidelity += prob / num_qubits
        return fidelity
