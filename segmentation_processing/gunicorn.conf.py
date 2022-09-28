"""Common configuration settings for Gunicorn servers."""

# pylint: disable=invalid-name

# h: remote address
# r: status line (e.g. `GET / HTTP/1.1`)
# s: status
# a: user agent
access_log_format = '%(h)s "%(r)s" %(s)s "%(a)s"'
log_level = "info"
logger_class = "server.GunicornLogger"
pythonpath = "."
