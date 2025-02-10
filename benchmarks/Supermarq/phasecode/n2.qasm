OPENQASM 2.0;
include "qelib1.inc";
gate state_preparation(param0) q0,q1 {  }
gate initialize(param0) q0,q1 { reset q0; reset q1; state_preparation(0) q0,q1; }
qreg q[3];
creg c[1];
creg meas[3];
initialize(0) q[0],q[2];
barrier q[0],q[1],q[2];
h q[0];
h q[2];
h q[0];
h q[1];
h q[2];
cz q[0],q[1];
cz q[2],q[1];
h q[1];
measure q[1] -> c[0];
reset q[1];
h q[0];
h q[2];
barrier q[0],q[1],q[2];
h q[0];
h q[2];
barrier q[0],q[1],q[2];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
