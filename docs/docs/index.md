# conspiracy founders documentation!

## Description

data science project regarding conspiracy founders

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `gsutil rsync` to recursively sync files in `data/` up to `gs://conspiracy_founders_bucket/data/`.
* `make sync_data_down` will use `gsutil rsync` to recursively sync files in `gs://conspiracy_founders_bucket/data/` to `data/`.


