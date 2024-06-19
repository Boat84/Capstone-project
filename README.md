# 





## Set up your Environment

Please make sure you have forked the repo and set up a new virtual environment.

The added [requirements file](requirements.txt) contains all libraries and dependencies we need to execute the NLP notebooks.

**`Note:`**

- If there are errors during environment setup, try removing the versions from the failing packages in the requirements file. M1 shizzle.
- In some cases it is necessary to install the **Rust** compiler for the transformers library.
- make sure to install **hdf5** if you haven't done it before.

 - Check the **rustup version**  by run the following commands:
    ```sh
    rustup --version
    ```
    If you haven't installed it yet, begin at `step_1`. Otherwise, proceed to `step_2`.


### **`macOS`** type the following commands : 

- `Step_1:` Update Homebrew and install **rustup** and **hdf5** by following commands:

    ```BASH
    brew install rustup
    rustup-init
    ```
    Then press ```1``` for the standard installation.
    
    Then we can go on to install hdf5:
    
    ```BASH
     brew install hdf5
    ```

  Restart Your Terminal and then check the **rustup version**  by running the following commands:
     ```sh
    rustup --version
    ```
 
- `Step_2:` Install the virtual environment and the required packages by following commands:

  > NOTE: for macOS with **M1, M2** chips (other than intel)
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements_M1.txt
    ```
  > NOTE: for macOS with **intel** chips
  ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    
### **`WindowsOS`** type the following commands :

- `Step_1:` Install **rustup**  by following :
  
  1. Visit the official Rust website: https://www.rust-lang.org/tools/install.
  2. Download and run the `rustup-init.exe` installer.
  3. Follow the on-screen instructions and choose the default options for a standard installation.
  4. Then press ```1``` for the standard installation.
 
     
    Then we can go on to install hdf5:

    ```sh
     choco upgrade chocolatey
     choco install hdf5
    ```
    Restart Your Terminal and then check the **rustup version**  by running the following commands:
  
     ```sh
    rustup --version
    ```

- `Step_2:` Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **`Note:`**
    If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:
    ```Bash
    python.exe -m pip install --upgrade pip
    ```
