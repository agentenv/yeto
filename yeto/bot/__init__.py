"""Training report / alert bot for Telegram and Lark (Feishu).

A read-only client of the yeto dashboard (``yeto dashboard serve``): it polls
``/api/view`` over HTTP, answers chat commands and pushes alerts on state
transitions. It never connects to training nodes. Run ``python -m yeto.bot``;
configuration is environment variables only (``yeto.bot.config``).
"""
