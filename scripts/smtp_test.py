import os
import smtplib
import ssl

host = os.environ.get("SMTP_HOST", "")
port = int(os.environ.get("SMTP_PORT", "587"))
user = os.environ.get("SMTP_USER", "")
pwd = os.environ.get("SMTP_PASS", "")
sender = os.environ.get("SMTP_FROM", user)
rcpt = os.environ.get("RECIPIENT", user)

missing = [k for k,v in [("SMTP_HOST", host), ("SMTP_USER", user), ("SMTP_PASS", pwd), ("SMTP_FROM", sender), ("RECIPIENT", rcpt)] if not v]
if missing:
    raise SystemExit(f"Missing required env vars: {', '.join(missing)}")

msg = f"From: {sender}\nTo: {rcpt}\nSubject: SMTP test\n\nThis is a test message.".encode("utf-8")
ctx = ssl.create_default_context()

print(f"Connecting to {host}:{port} as {user} -> {rcpt}")
with smtplib.SMTP(host, port, timeout=30) as s:
    s.starttls(context=ctx)
    s.login(user, pwd)
    s.sendmail(sender, [rcpt], msg)
print("SMTP send OK")
