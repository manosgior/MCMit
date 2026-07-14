create_ip -name fifo_generator -vendor xilinx.com -library ip -version 13.2 -module_name fifocore_8_64
set_property -dict {
CONFIG.Fifo_Implementation {Independent_Clocks_Block_RAM}
CONFIG.Input_Data_Width {8}
CONFIG.Input_Depth {128}
CONFIG.Output_Data_Width {64}
CONFIG.Output_Depth {16}
CONFIG.Reset_Type {Asynchronous_Reset}
CONFIG.Full_Flags_Reset_Value {1}
CONFIG.Valid_Flag {true}
CONFIG.Data_Count_Width {7}
CONFIG.Write_Data_Count_Width {7}
CONFIG.Read_Data_Count_Width {4}
CONFIG.Full_Threshold_Assert_Value {125}
CONFIG.Full_Threshold_Negate_Value {124}
CONFIG.Enable_Safety_Circuit {true}
} [get_ips fifocore_8_64]
#generate_target {instantiation_template} [get_files fifocore_8_64.xci]
#generate_target all [get_files  fifocore_8_64.xci]
#export_ip_user_files -of_objects [get_files fifocore_8_64.xci] -no_script -sync -force -quiet
#create_ip_run [get_files -of_objects [get_fileset sources_1] fifocore_8_64.xci]
