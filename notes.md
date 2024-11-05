## Step2

18 October 2024

### Install databricks cli
```
brew tap databricks/tap
brew install databricks
```

### Create Personal compute on primeai
```
https://adb-1713302240061567.7.azuredatabricks.net

```

### Authenticate
```
databricks auth login --configure-cluster --host https://adb-1713302240061567.7.azuredatabricks.net
```
This will create an entry in ~/.databrickscfg with auth type databricks-cli

### View profiles and token info
```
databricks auth profiles

databricks auth token --host https://adb-1713302240061567.7.azuredatabricks.net

```

##VS-Code

### Install Databricks plugin



## Build and copy to databricks
uv build

databricks auth login --host https://adb-1713302240061567.7.azuredatabricks.net

databricks fs cp dist/mlops_with_databricks-0.0.1-py3-none-any.whl  dbfs:/Volumes/heiaepgah71pwedmld01001/lj_loan_prediction/packages/ --profile primeai --overwrite


## Run precommit

Install precommit

uv pip install pre-commit


```
uv run pre-commit run --all-files
```

Also modify .github/workflows/ci.yml
