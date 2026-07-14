create_ip -name xadc_wiz -vendor xilinx.com -library ip -version 3.3 -module_name xadc_qubic
set_property -dict {
CONFIG.INTERFACE_SELECTION {ENABLE_DRP} 
CONFIG.DCLK_FREQUENCY {96} 
CONFIG.ENABLE_CALIBRATION_AVERAGING {false} 
CONFIG.OT_ALARM {false} 
CONFIG.USER_TEMP_ALARM {false} 
CONFIG.VCCINT_ALARM {false} 
CONFIG.VCCAUX_ALARM {false} 
CONFIG.ADC_OFFSET_AND_GAIN_CALIBRATION {false} 
CONFIG.SENSOR_OFFSET_AND_GAIN_CALIBRATION {false} 
CONFIG.XADC_STARUP_SELECTION {channel_sequencer} 
CONFIG.CHANNEL_ENABLE_TEMPERATURE {true} 
CONFIG.CHANNEL_ENABLE_VP_VN {false} 
CONFIG.CHANNEL_ENABLE_VAUXP4_VAUXN4 {true} 
CONFIG.CHANNEL_ENABLE_VAUXP12_VAUXN12 {true} 
CONFIG.SEQUENCER_MODE {Continuous} 
CONFIG.EXTERNAL_MUX_CHANNEL {VP_VN} 
CONFIG.SINGLE_CHANNEL_SELECTION {TEMPERATURE}
CONFIG.ADC_CONVERSION_RATE {1000}
} [get_ips xadc_qubic]
set_property -dict {
CONFIG.SIM_FILE_SEL {Relative_path}
CONFIG.SIM_FILE_REL_PATH {xadcsim/}
CONFIG.SIM_FILE_NAME {xadcsim}
} [get_ips xadc_qubic]
generate_target {instantiation_template} [get_files xadc_qubic.xci]
generate_target all [get_files  xadc_qubic.xci]
export_ip_user_files -of_objects [get_files xadc_qubic.xci] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] xadc_qubic.xci]
#generate_target all [get_files  xadc_qubic.xci]
#generate_target simulation [get_files  xadc_qubic.xci]
#CONFIG.SIM_FILE_SEL {Relative_path}
#CONFIG.SIM_FILE_REL_PATH {../../../../qubic.srcs/sources_1/ip/xadc_qubic/}
#CONFIG.SIM_MONITOR_FILE {../../../../../design.txt}
#CONFIG.SIM_FILE_REL_PATH {"../../../../../"}
#CONFIG.SIM_FILE_NAME {../../../../../designtest}
#CONFIG.SIM_FILE_NAME {/home/ghuang/work/LBNL/qubic/qubic/gh/designtest}
