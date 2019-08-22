### Docker image to use

jamesmcclain/aws-batch-ml:latest

## Here are the mappings from NLCD to supercategory:
```python
# simplified
nlcd_map = { 11:1, 12:2, 21:3, 22:3, 23:3, 24:3, 31:4, 41:5, 42:5, 43:5, 51:6, 52:6, 71:7, 72:7, 73:7, 74:7, 81:8, 82:8, 90:9, 95:9}
# extra simplified
nlcd_map = { 11:0, 12:0, 21:2, 22:2, 23:2, 24:2, 31:0, 41:3, 42:3, 43:3, 51:0, 52:0, 71:0, 72:0, 73:0, 74:0, 81:4, 82:4, 90:0, 95:0}
```
