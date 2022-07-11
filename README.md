# CapSense: A Real-Time Capacitive Sensor Simulation Framework for Physical Human-Robot Interaction

**CapSense** is a real-time capacitive sensor simulation framework aimed for use in robotic applications for future applications in sim-to-real transfer learning using proximity sensing modalities. The framework allows simulation of capacitive proximity sensors with different configuration parameters in different scenarios embedded in robot simulators such as *(i)* CoppeliaSim, *(ii)* Unity3D and *(iii*) PyBullet.

### Abstract

This article presents CapSense, a real-time open-source capacitive sensor simulation framework for robotic applications. CapSense provides raw data of capacitive proximity sensors based on a fast and efficient 3D FEM (Finite-Element-Method) implementation. The proposed framework is interfaced to off-the-shelf robot and physics simulation environments to couple dynamic interaction of the environment with an electrostatic solver for capacitance computation in real-time. The FEM method proposed in this article relies on a static tetrahedral mesh of the sensor surrounding without a-posteriori re-meshing and achieves high update rates by an adaptive update step. CapSense is flexible due to various configuration parameters (i.e. number, size, shape and location of electrodes) and serves as a platform for investigation of capacitive sensors in robotic applications. By using the proposed framework, researchers can simulate capacitive sensors in different scenarios and investigate these sensors and their configuration prior to installation and fabrication of real hardware. The proposed framework opens new research opportunities via sim-to-real transfer of capacitive sensing. The simulation approach is validated by comparing real-world results of different scenarios with simulation results. In order to showcase the benefits of CapSense in physical Human-Robot-Interaction (pHRI), the framework is evaluated in a robotic healthcare scenario.

### Main features

* Generation of capacitive sensor **raw data** in real-time
* Fully customizable capacitive sensor topologies based on STL files

A paper describing **CapSense** has been accepted to the IEEE Robotics and Automation Letters (RA-L) journal with presentation at IROS 2022 in Kyoto, Japan.

**NOTE:** We are continuosly working to improve and expand the CapSense framework. 


