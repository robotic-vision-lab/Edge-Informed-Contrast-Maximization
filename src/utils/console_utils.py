from termcolor import colored



def bf(text):
    """Returns bold text to print on terminal
    Args:
        text (string): bolded text
    """
    return colored(text, attrs=['bold'])


def df(text):
    """Returns dark text to print on terminal
    Args:
        text (string): bolded text
    """
    return colored(text, attrs=['dark'])


def r(text):
    """Returns red text to print on terminal
    Args:
        text (string): bolded text
    """
    return colored(text, 'red')


def m(text):
    """Returns magenta text to print on terminal
    Args:
        text (string): bolded text
    """
    return colored(text, 'magenta')


def g(text):
    """Returns green text to print on terminal
    Args:
        text (string): bolded text
    """
    return colored(text, 'green')


def y(text):
    """Returns yellow text to print on terminal
    Args:
        text (string): bolded text
    """
    return colored(text, 'yellow', attrs=['dark'])


def b(text):
    """Returns blue text to print on terminal
    Args:
        text (string): bolded text
    """
    return colored(text, 'blue')


def c(text):
    """Returns cyan text to print on terminal
    Args:
        text (string): bolded text
    """
    return colored(text, 'cyan')


def rb(text):
    """Returns bold red text to print on terminal
    Args:
        text (string): bolded text
    """
    return colored(text, 'red', attrs=['bold'])


def mb(text):
    """Returns bold magenta text to print on terminal
    Args:
        text (string): bolded text
    """
    return colored(text, 'magenta', attrs=['bold'])


def gb(text):
    """Returns bold green text to print on terminal
    Args:
        text (string): bolded text
    """
    return colored(text, 'green', attrs=['bold'])


def yb(text):
    """Returns bold yellow text to print on terminal
    Args:
        text (string): bolded text
    """
    return colored(text, 'yellow', attrs=['bold'])


def bb(text):
    """Returns bold blue text to print on terminal
    Args:
        text (string): bolded text
    """
    return colored(text, 'blue', attrs=['bold'])


def cb(text):
    """Returns bold cyan text to print on terminal
    Args:
        text (string): bolded text
    """
    return colored(text, 'cyan', attrs=['bold'])
