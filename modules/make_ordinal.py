def ordinal_suffix(task_id: int) -> str:
    """
    Generate the ordinal suffix for a given task ID.

    Parameters:
    - task_id (int): The task ID for which the ordinal suffix is to be generated.

    Returns:
    - str: The ordinal suffix corresponding to the task ID.

    Examples:
    >>> ordinal_suffix(0)
    'th'

    >>> ordinal_suffix(1)
    'st'

    >>> ordinal_suffix(11)
    'th'

    >>> ordinal_suffix(23)
    'rd'
    """
    task_id = str(task_id)
    if task_id.endswith(('11','12','13')):
        return 'th'
    elif task_id.endswith('1'):
        return 'st'
    elif task_id.endswith('2'):
        return 'nd'
    elif task_id.endswith('3'):
        return 'rd'
    else:
        return 'th'
