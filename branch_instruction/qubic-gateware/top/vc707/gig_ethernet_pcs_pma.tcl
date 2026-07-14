create_ip -name gig_ethernet_pcs_pma -vendor xilinx.com -library ip -module_name gig_ethernet_pcs_pma_0
set_property -dict {
CONFIG.Standard {SGMII}
CONFIG.Physical_Interface {Transceiver}
CONFIG.Management_Interface {true}
CONFIG.Ext_Management_Interface {false}
CONFIG.Auto_Negotiation {false}
} [get_ips gig_ethernet_pcs_pma_0]
#generate_target {instantiation_template} [get_files gig_ethernet_pcs_pma_0.xci]
#generate_target all [get_files  gig_ethernet_pcs_pma_0.xci]
#CONFIG.SGMII_PHY_Mode {true}
