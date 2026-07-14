# QubiC 2.0 Gateware

The QubiC 2.0 gateware is written primarily in Verilog and SystemVerilog, and synthesized using Xilinx Vivado 2022.1. Currently, only the ZCU216 RFSoC hardware platform is supported. 

## Configuring and Building

QubiC 2.0 builds can be configured with a variety of qubit drive (both RF and DC) and multiplexed readout channels. Builds are configured using `yaml`; an example top-level configuration file can be found in `top/dsp_config.yaml`. Existing builds can be found in various directories under `top`; e.g. ZCU216_14_2. Note that only config(s) + testbenches are version controlled; actual build outputs can be found in our [Releases](https://gitlab.com/LBL-QubiC/gateware/-/releases) page.

To initialize, configure, and synthesize a new build:

  1. `cd` into `top`, then run `./initialize_build.sh <build name>`. This will create a directory with  name `<build name>` and copy the associated config file + basic testbench into this directory.
  2. `cd` into `<build name>` and edit configuration file `dsp_config.yaml`.
  3. Run `./configure_build.sh <config file name>`. `<config file name>` will generally be `dsp_config.yaml`, unless you've created additional config files in the build directory.
  4. Step (3) will create a directory `build_<commit hash>_<timestamp>`. run:

    cd build_<commit hash>_<timestamp>
    make pre
    
This will generate all of the source files required for Vivado to synthesize the gateware.

  5. Run `make` to start the synthesis. For this to work, `vivado` must be in the system path and point to a valid installation. QubiC gateware builds are done using version 2022.1; later versions might work, but support for this is untested.

If you're only configuring an existing build, (optionally) edit (or create a new) `dsp_config.yaml` file in `top/<build name>`, then run steps 3-5 above.

### Should I initialize a new build configuration or modify an existing one?

Generally, if you're only making simple parameter changes (for example interpolation ratio or memory depth), you should modify an existing build. Note that multiple configuration files can be created and version controlled in the `<build name>` directory; and the config used for a particular configuration is saved in `build_<commit hash>_<timestamp>`. However, if you're making major changes (e.g. multiplexing factor or channel allocation), initializing a new `<build name>` is recommended. There are no hard rules here, just use your best judgement.

## Testbench

Each time a new build configuration is initialized, a basic cocotb test bench is created in `<build name>/cocotb`. To configure source files for the testbench, run:

    ./configure_build.sh <config file name> sim
    cd sim
    make pre

To run the testbench:

    cd cocotb
    make

The testbench comes with a simple RF pulse test; it is up to the user to write more detailed tests for specific builds. Note that testbench support is *experimental* and not guaranteed to work for user-initialized builds.

## Packaging a build for release/testing

After running the [configuring and building](#configuring-and-building) steps above, the resulting build outputs can be packaged for testing on hardware and/or release. To do this, navigate to the build directory (`build_<commit hash>_<timestamp>`; same directory as configuring and building), then run:

    make release

This will package the necessary outputs for hardware testing in `<commit_hash>.tar.gz`, including the `.xsa` file and necessary memory/register configs. Also included are the `channel_config.json` necessary for compiling circuits, and `channel_map.txt` which contains the mapping between signal gen channels defined in `dsp_config.yaml` and `channel_config.json` and physical ADC/DAC channels. For information on how to test a new build on hardware see the [getting started guide](https://gitlab.com/LBL-QubiC/software/-/wikis/Getting-Started-with-QubiC-2.0-on-the-ZCU216).

The entire build directory is also archived, which is useful for e.g. accessing the vivado project and build logs. The [Releases](https://gitlab.com/LBL-QubiC/gateware/-/releases) page contains both of these archives for all released builds.
