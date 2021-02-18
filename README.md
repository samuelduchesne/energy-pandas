[![PyPI version fury.io](https://badge.fury.io/py/energy-pandas.svg)](https://pypi.python.org/pypi/energy-pandas/)
[![codecov](https://codecov.io/gh/samuelduchesne/energy-pandas/branch/master/graph/badge.svg?token=kY9pzjlDZJ)](https://codecov.io/gh/samuelduchesne/energy-pandas)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/energy-pandas.svg)](https://pypi.python.org/pypi/energy-pandas/)

# energy-pandas

An extension of pandas to deal with building related time series such as temperature, load and energy.

## Installation

```python
pip install energy-pandas
```

## Usage

```python
from energy_pandas import EnergyDataFrame, EnergySeries

edf = EnergyDataFrame(
    {
        "temp": EnergySeries(range(0,100), units="degC"),
        "q_heat": EnergySeries(range(0,100), units="W"), 
        "q_cool": EnergySeries(range(0,100), units="W")
    },
    name="Zone 1"
)
edf.units
{'temp': <Unit('degree_Celsius')>, 'q_heat': <Unit('watt')>, 'q_cool': <Unit('watt')>}
```
