from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

service = QiskitRuntimeService()
job_id = 'd40t35cv6o9s73d07o10'

job = service.job(job_id)

print(job)
results = job.result()

pubs = job.inputs['pubs'] 
circuits = [pub[0] for pub in pubs]

print(circuits[0])

for r in results:
    print(r.data.meas.get_counts())
    exit()