import json
import os
import smtplib
import ssl
import urllib.request
from email.message import EmailMessage


def _env_or_value(value, env_key, default=None):
    if value not in (None, ""):
        return value
    return os.getenv(env_key, default)


def _format_message(
    *,
    run_name,
    curr_iter,
    epoch,
    train_loss,
    val_loss,
    val_pp,
    val_acc,
    lr,
    iter_dt,
):
    lines = [
        f"run: {run_name}",
        f"iter: {curr_iter}",
        f"epoch: {epoch:.3f}",
    ]
    if train_loss is not None:
        lines.append(f"train_loss: {train_loss:.4f}")
    if lr is not None:
        lines.append(f"lr: {lr:.6g}")
    if iter_dt is not None:
        lines.append(f"iter_dt: {iter_dt:.4f}s")
    if val_loss is not None:
        lines.append(f"val_loss: {val_loss:.4f}")
    if val_pp is not None:
        lines.append(f"val_pp: {val_pp:.4f}")
    if val_acc is not None:
        lines.append(f"val_acc: {val_acc:.4f}")
    return "\n".join(lines)


def _send_email(cfg, subject, body):
    host = _env_or_value(cfg.notify_smtp_host, "SMTP_HOST")
    port = int(_env_or_value(cfg.notify_smtp_port, "SMTP_PORT", "587"))
    user = _env_or_value(cfg.notify_smtp_user, "SMTP_USER")
    password = _env_or_value(cfg.notify_smtp_pass, "SMTP_PASS")
    from_addr = _env_or_value(cfg.notify_email_from, "SMTP_FROM", user)
    to_addr = _env_or_value(cfg.notify_email_to, "SMTP_TO")

    if not host or not from_addr or not to_addr:
        print(
            "Notify: email is not configured (missing SMTP_HOST/SMTP_FROM/SMTP_TO)."
        )
        return False

    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(host, port) as server:
            server.starttls(context=context)
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        return True
    except Exception as exc:
        print(f"Notify: email failed: {exc}")
        return False


def _send_webhook(cfg, subject, body, payload):
    url = _env_or_value(cfg.notify_webhook, "NOTIFY_WEBHOOK_URL")
    if not url:
        print("Notify: webhook is not configured (missing NOTIFY_WEBHOOK_URL).")
        return False

    data = json.dumps(
        {
            "subject": subject,
            "text": body,
            **payload,
        }
    ).encode("utf-8")

    try:
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as _:
            pass
        return True
    except Exception as exc:
        print(f"Notify: webhook failed: {exc}")
    return False


def _send_pushplus(cfg, subject, body):
    token = _env_or_value(cfg.notify_pushplus_token, "PUSHPLUS_TOKEN")
    topic = _env_or_value(cfg.notify_pushplus_topic, "PUSHPLUS_TOPIC")
    template = _env_or_value(cfg.notify_pushplus_template, "PUSHPLUS_TEMPLATE", "txt")
    url = _env_or_value(
        cfg.notify_pushplus_url, "PUSHPLUS_URL", "http://www.pushplus.plus/send"
    )

    if not token:
        print("Notify: PushPlus is not configured (missing PUSHPLUS_TOKEN).")
        return False

    payload = {
        "token": token,
        "title": subject,
        "content": body,
        "template": template,
    }
    if topic:
        payload["topic"] = topic

    data = json.dumps(payload).encode("utf-8")

    try:
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as _:
            pass
        return True
    except Exception as exc:
        print(f"Notify: PushPlus failed: {exc}")
        return False


def maybe_notify(
    cfg,
    *,
    curr_iter,
    epoch,
    train_loss=None,
    val_loss=None,
    val_pp=None,
    val_acc=None,
    lr=None,
    iter_dt=None,
    run_name="experiment",
):
    if not cfg.notify_interval or cfg.notify_interval <= 0:
        return False
    if curr_iter % cfg.notify_interval != 0:
        return False

    subject = f"[{run_name}] iter {curr_iter}"
    body = _format_message(
        run_name=run_name,
        curr_iter=curr_iter,
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        val_pp=val_pp,
        val_acc=val_acc,
        lr=lr,
        iter_dt=iter_dt,
    )

    method = (cfg.notify_method or "stdout").lower()
    payload = {
        "iter": curr_iter,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_pp": val_pp,
        "val_acc": val_acc,
        "lr": lr,
        "iter_dt": iter_dt,
        "run_name": run_name,
    }

    if method == "stdout":
        print(f"Notify:\n{body}")
        return True
    if method == "email":
        return _send_email(cfg, subject, body)
    if method == "webhook":
        return _send_webhook(cfg, subject, body, payload)
    if method == "pushplus":
        return _send_pushplus(cfg, subject, body)

    print(f"Notify: unknown method '{cfg.notify_method}', skipping.")
    return False
