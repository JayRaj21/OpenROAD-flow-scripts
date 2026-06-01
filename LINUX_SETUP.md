# Linux Setup Guide for OpenROAD-flow-scripts

Tested on **Ubuntu 26.04 LTS**. Should also work on Ubuntu 22.04/24.04.

---

## Prerequisites

- 64-bit Linux (Ubuntu 22.04, 24.04, or 26.04 recommended)
- At least 8 GB RAM, 20 GB free disk space
- Internet connection for pulling the Docker image (~4 GB)

---

## Step 1 — Clone the Repository

```
git clone https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git
cd OpenROAD-flow-scripts
```

> **Note:** Do not clone recursively unless you intend to build from source. The Docker-based flow does not require the submodules.

---

## Step 2 — Install Docker

```
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
```

### Allow running Docker without sudo

```
sudo usermod -aG docker $USER
sudo apt install -y util-linux-extra
newgrp docker
```

> **Note:** For the group change to persist across reboots, log out and back in to your desktop session.

### Verify Docker is working

```
docker run --rm hello-world
```

---

## Step 3 — Pull the ORFS Docker Image

```
docker pull openroad/orfs:latest
```

This image bundles OpenROAD, Yosys, and KLayout with all dependencies. No further tool installation is required.

---

## Step 4 — Run a Test Design

The repository includes a pre-configured 4-bit adder design for smoke-testing the flow.

```
bash flow/run_adder4.sh
```

This runs the full RTL-to-GDSII flow (synthesis → floorplan → placement → CTS → routing → finishing). It takes approximately 5–15 minutes on a typical laptop.

Output files land in `flow/results/nangate45/adder4/base/`:

| File | Description |
|---|---|
| `6_final.gds` | Final GDSII layout |
| `6_final.odb` | OpenROAD database (viewable in GUI) |
| `6_final.def` | Final DEF (physical layout) |

Timing and DRC reports are in `flow/reports/nangate45/adder4/base/`.

---

## Step 5 — Validate the Design (Optional)

### DRC (Design Rule Check)

```
bash flow/drc_adder4.sh
```

### Open the GUI

```
bash flow/gui_adder4.sh
```

> **Note:** GUI requires a display. If running over SSH, set up X11 forwarding or use a VNC session.

---

## Running Your Own Design

To run a custom Verilog design:

1. Place your `.v` file in `flow/designs/src/<design_name>/`
2. Create `flow/designs/nangate45/<design_name>/config.mk` and `constraint.sdc` (see `flow/designs/nangate45/adder4/` as a reference)
3. Run the flow:

```
util/docker_shell make DESIGN_CONFIG=/work/designs/nangate45/<design_name>/config.mk DESIGN_HOME=/work/designs
```

### config.mk template

```makefile
export DESIGN_NAME = <design_name>
export PLATFORM    = nangate45

export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NAME)/<top_module>.v
export SDC_FILE      = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NAME)/constraint.sdc

export CORE_UTILIZATION  = 40
export PLACE_DENSITY_LB_ADDON = 0.10
export TNS_END_PERCENT   = 100
```

> **Tip:** If your design is very small (< 50 standard cells), lower `CORE_UTILIZATION` to 3–5% to avoid PDN errors. The die must be wide enough to fit power grid straps.

---

## Troubleshooting

### `permission denied while trying to connect to the Docker socket`

The docker group change has not taken effect. Run:

```
newgrp docker
```

Or log out and back in to your desktop session.

### `PLATFORM variable not set`

The `DESIGN_CONFIG` path must be an absolute path inside the container. Use `/work/designs/...` not `designs/...`:

```
util/docker_shell make DESIGN_CONFIG=/work/designs/nangate45/<design>/config.mk DESIGN_HOME=/work/designs
```

### `Insufficient width to add straps on layer metal4`

The die is too small for the power grid. Lower `CORE_UTILIZATION` in `config.mk` (try 3–5%) and re-run.

### `newgrp: command not found`

```
sudo apt install -y util-linux-extra
```

---

## Available Platforms

| Platform | Node | Open Source |
|---|---|---|
| `nangate45` | 45 nm (academic) | Yes |
| `sky130hd` | 130 nm | Yes |
| `sky130hs` | 130 nm (high speed) | Yes |
| `asap7` | 7 nm (academic) | Yes |
| `ihp-sg13g2` | 130 nm (IHP) | Yes |
| `gf180` | 180 nm | Yes |
