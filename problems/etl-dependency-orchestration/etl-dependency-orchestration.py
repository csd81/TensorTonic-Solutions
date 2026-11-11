import heapq


def schedule_pipeline(tasks: list, resource_budget: int) -> list:
    """
    Returns a list of schedule dictionaries.
    """
    if not tasks:
        return []

    task_map = {t["name"]: t for t in tasks}
    unscheduled = set(task_map.keys())
    completed = set()

    # Min-heap storing (end_time, task_name, resources_used)
    running = []
    available_resources = resource_budget
    current_time = 0
    schedule = []

    while unscheduled:
        # 1. Complete every running task whose end time has been reached
        while running and running[0][0] <= current_time:
            end_time, finished_name, freed_resources = heapq.heappop(running)
            completed.add(finished_name)
            available_resources += freed_resources

        # 2. Collect unscheduled tasks whose dependencies are all complete, sorted alphabetically
        ready_tasks = sorted(
            name
            for name in unscheduled
            if all(dep in completed for dep in task_map[name]["depends_on"])
        )

        # 3. Greedily start each ready task that fits the remaining resource budget
        for name in ready_tasks:
            req_resources = task_map[name]["resources"]
            if req_resources <= available_resources:
                available_resources -= req_resources
                unscheduled.remove(name)
                schedule.append({"task_name": name, "start_time": current_time})
                end_time = current_time + task_map[name]["duration"]
                heapq.heappush(running, (end_time, name, req_resources))

        # 4. Advance time to the next running-task completion if unscheduled tasks remain
        if unscheduled and running:
            current_time = running[0][0]

    # Sort output by start time, then alphabetically by task name
    schedule.sort(key=lambda item: (item["start_time"], item["task_name"]))
    return schedule
