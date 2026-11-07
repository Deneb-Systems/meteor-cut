# Security

Please report security problems privately to **contact@deneb-systems.com**, not in a public issue. We aim to reply
within a week.

## What to expect from the upload page

`meteor-cut serve` is meant for one person on their own computer:

- It listens on `127.0.0.1` unless told otherwise, and warns when given another address.
- It has no accounts: anyone who can reach it can upload videos and download the results of any job they know the ID
  of. Do not expose it to a network you do not trust.
- Uploads are limited in size (20 GB by default, `--max-upload-gb`) and type (video file extensions), processed one at
  a time, and deleted with the rest of its working folder when the server stops.
- It runs on Flask's built-in server, which is not built for public traffic.
