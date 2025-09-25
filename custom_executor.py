from qiskit_aer import AerSimulator
from qiskit.result import Result
from qiskit import QuantumCircuit
from typing import List, Dict, Union
import numpy as np

class CustomCircuitExecutor:
    def __init__(self):
        self.backend = AerSimulator()
        
    def _fully_decompose(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Recursively decompose a circuit until it only contains basic gates.
        """
        decomposed = circuit.decompose()
        while any(op.name.startswith('circuit') for op, _, _ in decomposed.data):
            decomposed = decomposed.decompose()
        return decomposed
        
    def run(self, 
            circuits: List[QuantumCircuit],
            parameter_values: List[List[float]], 
            shots: int = 8192) -> 'CircuitResults':
        # Get the template circuit and decompose it once
        template_circuit = self._fully_decompose(circuits[0])
        circuit_params = template_circuit.parameters
        
        # Bind parameters and execute circuits
        bound_circuits = []
        for params in parameter_values:
            param_dict = dict(zip(circuit_params, params))
            
            bound_circuit = template_circuit.assign_parameters(param_dict)
            bound_circuits.append(bound_circuit)
            
        job = self.backend.run(bound_circuits, shots=shots)
        raw_results = job.result()
            
        return CircuitResults(raw_results)

class CircuitResults:
    def __init__(self, qiskit_result: Result):
        self.raw_result = qiskit_result
        self._process_results()
        
    def _process_results(self):
        """Process raw results into required format."""
        self.quasi_dists = []
        for i in range(len(self.raw_result.results)):
            counts = self.raw_result.get_counts(i)
            # Convert counts to quasi-distribution format
            total_shots = sum(counts.values())
            quasi_dist = {int(key, 2): value/total_shots 
                         for key, value in counts.items()}
            self.quasi_dists.append(quasi_dist)
    
    def result(self) -> 'CircuitResults':
        return self