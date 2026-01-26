"""Command-line interface using Typer."""

from typing import Optional
from uuid import UUID

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shorts_engine import __version__
from shorts_engine.logging import setup_logging

# Setup logging
setup_logging()

app = typer.Typer(
    name="shorts-engine",
    help="AI Shorts Engine - Closed-loop video generation CLI",
    add_completion=False,
)

# Subcommand groups
shorts_app = typer.Typer(help="Video shorts creation commands")
projects_app = typer.Typer(help="Project management commands")
accounts_app = typer.Typer(help="Platform account management commands")
ingest_app = typer.Typer(help="Analytics and comments ingestion commands")
app.add_typer(shorts_app, name="shorts")
app.add_typer(projects_app, name="projects")
app.add_typer(accounts_app, name="accounts")
app.add_typer(ingest_app, name="ingest")

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"AI Shorts Engine v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """AI Shorts Engine - Generate, publish, and optimize short-form videos."""
    pass


@app.command()
def smoke() -> None:
    """Run a smoke test to verify the pipeline is working."""
    console.print("[bold blue]Running smoke test...[/bold blue]")

    try:
        from shorts_engine.jobs.tasks import smoke_test_task

        # Run synchronously for CLI
        result = smoke_test_task.delay()
        console.print(f"[dim]Task ID: {result.id}[/dim]")

        # Wait for result with timeout
        console.print("[dim]Waiting for result...[/dim]")
        task_result = result.get(timeout=60)

        if task_result.get("success"):
            console.print("[bold green]✓ Smoke test passed![/bold green]")

            # Show stages table
            table = Table(title="Pipeline Stages")
            table.add_column("Stage", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details")

            stages = task_result.get("stages", {})
            for stage, info in stages.items():
                status = "✓" if info.get("success") else "✗"
                details = ", ".join(f"{k}={v}" for k, v in info.items() if k != "success")
                table.add_row(stage, status, details[:50])

            console.print(table)
        else:
            console.print(f"[bold red]✗ Smoke test failed: {task_result.get('error')}[/bold red]")
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="The prompt for video generation"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Video title"),
    duration: int = typer.Option(60, "--duration", "-d", help="Duration in seconds"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for completion"),
) -> None:
    """Generate a video from a prompt."""
    console.print(f"[bold blue]Generating video...[/bold blue]")
    console.print(f"[dim]Prompt: {prompt[:100]}...[/dim]" if len(prompt) > 100 else f"[dim]Prompt: {prompt}[/dim]")

    try:
        from shorts_engine.jobs.tasks import generate_video_task

        result = generate_video_task.delay(
            prompt=prompt,
            title=title,
            duration_seconds=duration,
        )
        console.print(f"[green]Task enqueued: {result.id}[/green]")

        if wait:
            console.print("[dim]Waiting for result...[/dim]")
            task_result = result.get(timeout=300)

            if task_result.get("success"):
                console.print("[bold green]✓ Video generated successfully![/bold green]")
                console.print(f"Title: {task_result.get('title')}")
            else:
                console.print(f"[bold red]✗ Generation failed: {task_result.get('error')}[/bold red]")
                raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def status(
    task_id: str = typer.Argument(..., help="The task ID to check"),
) -> None:
    """Check the status of a job."""
    try:
        from celery.result import AsyncResult

        from shorts_engine.worker import celery_app

        result = AsyncResult(task_id, app=celery_app)

        table = Table(title="Job Status")
        table.add_column("Field", style="cyan")
        table.add_column("Value")

        table.add_row("Task ID", task_id)
        table.add_row("Status", result.state)

        if result.state == "SUCCESS":
            table.add_row("Result", str(result.result)[:200])
        elif result.state == "FAILURE":
            table.add_row("Error", str(result.result))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def health() -> None:
    """Check the health of all services."""
    import httpx

    from shorts_engine.config import settings

    url = f"http://{settings.api_host}:{settings.api_port}/health/ready"

    try:
        response = httpx.get(url, timeout=10)
        data = response.json()

        table = Table(title="Service Health")
        table.add_column("Component", style="cyan")
        table.add_column("Status")

        table.add_row("Database", "✓" if data.get("database") else "✗")
        table.add_row("Redis", "✓" if data.get("redis") else "✗")

        for component, healthy in data.get("components", {}).items():
            table.add_row(component, "✓" if healthy else "✗")

        console.print(table)

        if data.get("ready"):
            console.print("[bold green]All services healthy![/bold green]")
        else:
            console.print("[bold yellow]Some services unhealthy[/bold yellow]")
            raise typer.Exit(code=1)

    except httpx.RequestError as e:
        console.print(f"[bold red]Cannot connect to API: {e}[/bold red]")
        console.print("[dim]Is the API server running?[/dim]")
        raise typer.Exit(code=1)


@app.command()
def worker() -> None:
    """Start a Celery worker (for development)."""
    console.print("[bold blue]Starting Celery worker...[/bold blue]")

    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "celery", "-A", "shorts_engine.worker", "worker", "--loglevel=info"],
        check=True,
    )


# =============================================================================
# SHORTS COMMANDS
# =============================================================================


@shorts_app.command("create")
def shorts_create(
    project: str = typer.Option(..., "--project", "-p", help="Project ID (UUID)"),
    idea: str = typer.Option(..., "--idea", "-i", help="One-paragraph video idea"),
    preset: str = typer.Option(
        "DARK_DYSTOPIAN_ANIME",
        "--preset",
        "-s",
        help="Style preset (DARK_DYSTOPIAN_ANIME, VIBRANT_MOTION_GRAPHICS, CINEMATIC_REALISM, SURREAL_DREAMSCAPE)",
    ),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for pipeline completion"),
) -> None:
    """Create a new short video from an idea.

    Enqueues a full video creation pipeline: plan -> generate scenes -> verify -> ready.

    Example:
        shorts-engine shorts create --project <uuid> --idea "A lone samurai walks through a ruined city" --preset DARK_DYSTOPIAN_ANIME
    """
    from shorts_engine.presets.styles import get_preset, get_preset_names

    # Validate project ID
    try:
        project_uuid = UUID(project)
    except ValueError:
        console.print(f"[bold red]Invalid project ID: {project}[/bold red]")
        console.print("[dim]Use 'shorts-engine projects list' to see available projects[/dim]")
        raise typer.Exit(code=1)

    # Validate preset
    preset_upper = preset.upper()
    if not get_preset(preset_upper):
        available = ", ".join(get_preset_names())
        console.print(f"[bold red]Unknown preset: {preset}[/bold red]")
        console.print(f"[dim]Available presets: {available}[/dim]")
        raise typer.Exit(code=1)

    # Verify project exists
    from shorts_engine.db.models import ProjectModel
    from shorts_engine.db.session import get_session_context

    with get_session_context() as session:
        project_record = session.get(ProjectModel, project_uuid)
        if not project_record:
            console.print(f"[bold red]Project not found: {project}[/bold red]")
            console.print("[dim]Use 'shorts-engine projects create' to create a new project[/dim]")
            raise typer.Exit(code=1)

        project_name = project_record.name

    console.print(Panel.fit(
        f"[bold]Creating Short Video[/bold]\n\n"
        f"[cyan]Project:[/cyan] {project_name}\n"
        f"[cyan]Preset:[/cyan] {preset_upper}\n"
        f"[cyan]Idea:[/cyan] {idea[:100]}{'...' if len(idea) > 100 else ''}",
        title="Video Creation Pipeline",
        border_style="blue",
    ))

    try:
        from shorts_engine.jobs.video_pipeline import run_full_pipeline_task

        # Enqueue the pipeline
        result = run_full_pipeline_task.delay(
            project_id=str(project_uuid),
            idea=idea,
            style_preset=preset_upper,
        )

        console.print(f"\n[green]Pipeline enqueued successfully![/green]")
        console.print(f"[dim]Task ID: {result.id}[/dim]")

        if wait:
            console.print("\n[dim]Waiting for pipeline completion...[/dim]")
            with console.status("[bold blue]Processing...", spinner="dots"):
                task_result = result.get(timeout=3600)  # 1 hour max

            if task_result.get("success"):
                console.print("\n[bold green]Pipeline completed successfully![/bold green]")

                # Show result details
                table = Table(title="Pipeline Result")
                table.add_column("Field", style="cyan")
                table.add_column("Value")

                table.add_row("Video Job ID", task_result.get("video_job_id", "N/A"))
                table.add_row("Pipeline Task ID", task_result.get("pipeline_task_id", "N/A"))
                table.add_row("Idempotency Key", task_result.get("idempotency_key", "N/A"))

                console.print(table)

                # Get final job details
                job_id = task_result.get("video_job_id")
                if job_id:
                    _show_job_details(job_id)
            else:
                console.print(f"\n[bold red]Pipeline failed![/bold red]")
                console.print(f"[red]{task_result.get('error', 'Unknown error')}[/red]")
                raise typer.Exit(code=1)
        else:
            console.print(f"\n[dim]Use 'shorts-engine shorts status {result.id}' to check progress[/dim]")
            console.print(f"[dim]Or 'shorts-engine shorts job <video_job_id>' to see job details[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


def _show_job_details(job_id: str) -> None:
    """Display detailed job information."""
    from shorts_engine.db.models import AssetModel, SceneModel, VideoJobModel
    from shorts_engine.db.session import get_session_context
    from sqlalchemy import select

    with get_session_context() as session:
        job = session.get(VideoJobModel, UUID(job_id))
        if not job:
            return

        console.print(f"\n[bold]Video Job Details[/bold]")
        console.print(f"[cyan]Title:[/cyan] {job.title or 'Untitled'}")
        console.print(f"[cyan]Description:[/cyan] {(job.description or 'N/A')[:200]}")
        console.print(f"[cyan]Status:[/cyan] {job.status}")
        console.print(f"[cyan]Stage:[/cyan] {job.stage}")

        # Show scenes
        scenes = session.execute(
            select(SceneModel).where(SceneModel.video_job_id == job.id).order_by(SceneModel.scene_number)
        ).scalars().all()

        if scenes:
            console.print(f"\n[bold]Scenes ({len(scenes)})[/bold]")
            scene_table = Table()
            scene_table.add_column("#", style="dim")
            scene_table.add_column("Caption Beat", style="cyan")
            scene_table.add_column("Duration", style="green")
            scene_table.add_column("Status")

            for scene in scenes:
                scene_table.add_row(
                    str(scene.scene_number),
                    scene.caption_beat or "-",
                    f"{scene.duration_seconds:.1f}s",
                    scene.status,
                )

            console.print(scene_table)

        # Show assets
        assets = session.execute(
            select(AssetModel).where(
                AssetModel.video_job_id == job.id,
                AssetModel.status == "ready",
            )
        ).scalars().all()

        if assets:
            console.print(f"\n[bold]Assets ({len(assets)})[/bold]")
            for asset in assets:
                url = asset.url or f"file://{asset.file_path}"
                console.print(f"  [dim]{asset.asset_type}:[/dim] {url[:80]}...")


@shorts_app.command("status")
def shorts_status(
    task_id: str = typer.Argument(..., help="Celery task ID to check"),
) -> None:
    """Check the status of a video creation pipeline task."""
    from celery.result import AsyncResult
    from shorts_engine.worker import celery_app

    result = AsyncResult(task_id, app=celery_app)

    table = Table(title="Pipeline Status")
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("Task ID", task_id)
    table.add_row("State", result.state)

    if result.state == "SUCCESS":
        task_result = result.result
        table.add_row("Success", str(task_result.get("success", False)))
        table.add_row("Video Job ID", task_result.get("video_job_id", "N/A"))
        if task_result.get("video_job_id"):
            console.print(table)
            _show_job_details(task_result["video_job_id"])
            return
    elif result.state == "FAILURE":
        table.add_row("Error", str(result.result))
    elif result.state == "PENDING":
        table.add_row("Info", "Task is pending or unknown")
    else:
        table.add_row("Info", result.info if result.info else "Processing...")

    console.print(table)


@shorts_app.command("job")
def shorts_job(
    job_id: str = typer.Argument(..., help="Video job ID (UUID)"),
) -> None:
    """Show detailed information about a video job."""
    try:
        job_uuid = UUID(job_id)
    except ValueError:
        console.print(f"[bold red]Invalid job ID: {job_id}[/bold red]")
        raise typer.Exit(code=1)

    _show_job_details(str(job_uuid))


@shorts_app.command("list")
def shorts_list(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project ID"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of jobs to show"),
) -> None:
    """List recent video jobs."""
    from shorts_engine.db.models import VideoJobModel
    from shorts_engine.db.session import get_session_context
    from sqlalchemy import select

    with get_session_context() as session:
        query = select(VideoJobModel).order_by(VideoJobModel.created_at.desc()).limit(limit)

        if project:
            try:
                project_uuid = UUID(project)
                query = query.where(VideoJobModel.project_id == project_uuid)
            except ValueError:
                console.print(f"[bold red]Invalid project ID: {project}[/bold red]")
                raise typer.Exit(code=1)

        jobs = session.execute(query).scalars().all()

        if not jobs:
            console.print("[dim]No video jobs found[/dim]")
            return

        table = Table(title="Video Jobs")
        table.add_column("ID", style="dim", no_wrap=True)
        table.add_column("Title", style="cyan")
        table.add_column("Preset")
        table.add_column("Status", style="green")
        table.add_column("Stage")
        table.add_column("Created")

        for job in jobs:
            table.add_row(
                str(job.id)[:8] + "...",
                (job.title or "Untitled")[:30],
                job.style_preset[:15],
                job.status,
                job.stage,
                job.created_at.strftime("%Y-%m-%d %H:%M") if job.created_at else "-",
            )

        console.print(table)


@shorts_app.command("render")
def shorts_render(
    job_id: str = typer.Option(..., "--job", "-j", help="Video job ID (UUID) to render"),
    voiceover: bool = typer.Option(True, "--voiceover/--no-voiceover", help="Generate voiceover"),
    captions: bool = typer.Option(True, "--captions/--no-captions", help="Burn in captions"),
    voice: Optional[str] = typer.Option(None, "--voice", "-v", help="Voice ID for voiceover"),
    music_url: Optional[str] = typer.Option(None, "--music", "-m", help="Background music URL"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for render completion"),
) -> None:
    """Render a final video from a completed job.

    Takes scene clips and composes them into a final MP4 with optional
    voiceover and burned-in captions.

    Example:
        shorts-engine shorts render --job <uuid> --wait
        shorts-engine shorts render --job <uuid> --no-voiceover --music https://...
    """
    from shorts_engine.db.models import VideoJobModel
    from shorts_engine.db.session import get_session_context

    # Validate job ID
    try:
        job_uuid = UUID(job_id)
    except ValueError:
        console.print(f"[bold red]Invalid job ID: {job_id}[/bold red]")
        raise typer.Exit(code=1)

    # Verify job exists and is ready for rendering
    with get_session_context() as session:
        job = session.get(VideoJobModel, job_uuid)
        if not job:
            console.print(f"[bold red]Video job not found: {job_id}[/bold red]")
            raise typer.Exit(code=1)

        # Check job stage
        valid_stages = ("ready", "planned", "generated", "verified", "ready_for_render")
        if job.stage not in valid_stages and job.status != "completed":
            console.print(f"[bold yellow]Warning: Job stage is '{job.stage}', status is '{job.status}'[/bold yellow]")
            console.print("[dim]The job may not have completed scene generation yet.[/dim]")

        job_title = job.title or "Untitled"

    console.print(Panel.fit(
        f"[bold]Rendering Final Video[/bold]\n\n"
        f"[cyan]Job:[/cyan] {job_title}\n"
        f"[cyan]Job ID:[/cyan] {job_id}\n"
        f"[cyan]Voiceover:[/cyan] {'Yes' if voiceover else 'No'}\n"
        f"[cyan]Captions:[/cyan] {'Yes' if captions else 'No'}\n"
        f"[cyan]Voice:[/cyan] {voice or 'default'}\n"
        f"[cyan]Music:[/cyan] {music_url or 'None'}",
        title="Render Pipeline",
        border_style="green",
    ))

    try:
        from shorts_engine.jobs.render_pipeline import run_render_pipeline_task

        # Enqueue the render pipeline
        result = run_render_pipeline_task.delay(
            video_job_id=str(job_uuid),
            include_voiceover=voiceover,
            include_captions=captions,
            voice_id=voice,
            background_music_url=music_url,
        )

        console.print(f"\n[green]Render pipeline enqueued successfully![/green]")
        console.print(f"[dim]Task ID: {result.id}[/dim]")

        if wait:
            console.print("\n[dim]Waiting for render completion...[/dim]")
            with console.status("[bold green]Rendering...", spinner="dots"):
                task_result = result.get(timeout=1800)  # 30 min max

            if task_result.get("success"):
                console.print("\n[bold green]Render completed successfully![/bold green]")

                # Show result details
                table = Table(title="Render Result")
                table.add_column("Field", style="cyan")
                table.add_column("Value")

                table.add_row("Video Job ID", task_result.get("video_job_id", "N/A"))
                table.add_row("Stage", task_result.get("stage", "N/A"))
                table.add_row("Final MP4 URL", task_result.get("final_mp4_url", "N/A"))

                console.print(table)

                # Show the final URL prominently
                final_url = task_result.get("final_mp4_url")
                if final_url:
                    console.print(f"\n[bold cyan]Final Video URL:[/bold cyan]")
                    console.print(f"  {final_url}")
            else:
                console.print(f"\n[bold red]Render failed![/bold red]")
                console.print(f"[red]{task_result.get('error', 'Unknown error')}[/red]")
                raise typer.Exit(code=1)
        else:
            console.print(f"\n[dim]Use 'shorts-engine shorts status {result.id}' to check progress[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


# =============================================================================
# PROJECTS COMMANDS
# =============================================================================


@projects_app.command("create")
def projects_create(
    name: str = typer.Option(..., "--name", "-n", help="Project name"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Project description"),
    preset: str = typer.Option(
        "DARK_DYSTOPIAN_ANIME",
        "--preset",
        "-s",
        help="Default style preset",
    ),
) -> None:
    """Create a new project (content brand/channel)."""
    from uuid import uuid4
    from shorts_engine.db.models import ProjectModel
    from shorts_engine.db.session import get_session_context
    from shorts_engine.presets.styles import get_preset, get_preset_names

    # Validate preset
    preset_upper = preset.upper()
    if not get_preset(preset_upper):
        available = ", ".join(get_preset_names())
        console.print(f"[bold red]Unknown preset: {preset}[/bold red]")
        console.print(f"[dim]Available presets: {available}[/dim]")
        raise typer.Exit(code=1)

    with get_session_context() as session:
        project = ProjectModel(
            id=uuid4(),
            name=name,
            description=description,
            default_style_preset=preset_upper,
            is_active=True,
        )
        session.add(project)
        session.commit()

        console.print(f"[bold green]Project created successfully![/bold green]")
        console.print(f"[cyan]ID:[/cyan] {project.id}")
        console.print(f"[cyan]Name:[/cyan] {project.name}")
        console.print(f"[cyan]Default Preset:[/cyan] {project.default_style_preset}")

        console.print(f"\n[dim]Create a short with:[/dim]")
        console.print(f"[dim]  shorts-engine shorts create --project {project.id} --idea \"Your idea here\"[/dim]")


@projects_app.command("list")
def projects_list() -> None:
    """List all projects."""
    from shorts_engine.db.models import ProjectModel
    from shorts_engine.db.session import get_session_context
    from sqlalchemy import select

    with get_session_context() as session:
        projects = session.execute(
            select(ProjectModel).order_by(ProjectModel.created_at.desc())
        ).scalars().all()

        if not projects:
            console.print("[dim]No projects found. Create one with 'shorts-engine projects create'[/dim]")
            return

        table = Table(title="Projects")
        table.add_column("ID", style="dim")
        table.add_column("Name", style="cyan")
        table.add_column("Default Preset")
        table.add_column("Active", style="green")
        table.add_column("Created")

        for project in projects:
            table.add_row(
                str(project.id),
                project.name,
                project.default_style_preset or "-",
                "Yes" if project.is_active else "No",
                project.created_at.strftime("%Y-%m-%d") if project.created_at else "-",
            )

        console.print(table)


@projects_app.command("show")
def projects_show(
    project_id: str = typer.Argument(..., help="Project ID (UUID)"),
) -> None:
    """Show details of a project."""
    from shorts_engine.db.models import ProjectModel, VideoJobModel
    from shorts_engine.db.session import get_session_context
    from sqlalchemy import func, select

    try:
        project_uuid = UUID(project_id)
    except ValueError:
        console.print(f"[bold red]Invalid project ID: {project_id}[/bold red]")
        raise typer.Exit(code=1)

    with get_session_context() as session:
        project = session.get(ProjectModel, project_uuid)
        if not project:
            console.print(f"[bold red]Project not found: {project_id}[/bold red]")
            raise typer.Exit(code=1)

        # Count jobs
        job_count = session.execute(
            select(func.count(VideoJobModel.id)).where(VideoJobModel.project_id == project_uuid)
        ).scalar()

        console.print(Panel.fit(
            f"[bold]{project.name}[/bold]\n\n"
            f"[cyan]ID:[/cyan] {project.id}\n"
            f"[cyan]Description:[/cyan] {project.description or 'N/A'}\n"
            f"[cyan]Default Preset:[/cyan] {project.default_style_preset or 'N/A'}\n"
            f"[cyan]Active:[/cyan] {'Yes' if project.is_active else 'No'}\n"
            f"[cyan]Video Jobs:[/cyan] {job_count}\n"
            f"[cyan]Created:[/cyan] {project.created_at.strftime('%Y-%m-%d %H:%M') if project.created_at else 'N/A'}",
            title="Project Details",
            border_style="blue",
        ))


# =============================================================================
# PRESETS COMMAND
# =============================================================================


@app.command()
def presets() -> None:
    """List available style presets."""
    from shorts_engine.presets.styles import PRESETS

    for name, preset in PRESETS.items():
        console.print(Panel.fit(
            f"[bold]{preset.display_name}[/bold]\n\n"
            f"{preset.description}\n\n"
            f"[cyan]Aspect Ratio:[/cyan] {preset.aspect_ratio.value}\n"
            f"[cyan]Scene Duration:[/cyan] {preset.default_duration_per_scene}s\n"
            f"[cyan]Camera Style:[/cyan] {preset.camera_style}\n\n"
            f"[dim]Style Tokens:[/dim] {', '.join(preset.style_tokens[:5])}...",
            title=name,
            border_style="cyan",
        ))
        console.print()


# =============================================================================
# ACCOUNTS COMMANDS
# =============================================================================


@accounts_app.command("connect")
def accounts_connect(
    platform: str = typer.Argument(..., help="Platform to connect (youtube)"),
    label: str = typer.Option(..., "--label", "-l", help="User-friendly label for this account"),
    device_flow: bool = typer.Option(True, "--device-flow/--browser", help="Use device flow (recommended) or browser redirect"),
) -> None:
    """Connect a platform account using OAuth.

    This will open a browser window or display a code for authorization.

    Example:
        shorts-engine accounts connect youtube --label "Main Channel"
    """
    platform = platform.lower()

    if platform != "youtube":
        console.print(f"[bold red]Unsupported platform: {platform}[/bold red]")
        console.print("[dim]Currently supported: youtube[/dim]")
        raise typer.Exit(code=1)

    console.print(Panel.fit(
        f"[bold]Connecting {platform.title()} Account[/bold]\n\n"
        f"[cyan]Label:[/cyan] {label}\n"
        f"[cyan]Auth Method:[/cyan] {'Device Flow' if device_flow else 'Browser Redirect'}\n\n"
        "[dim]You will be prompted to authorize access to your account.[/dim]",
        title="OAuth Setup",
        border_style="blue",
    ))

    try:
        from shorts_engine.db.session import get_session_context
        from shorts_engine.services.accounts import connect_youtube_account, AccountError

        with get_session_context() as session:
            account = connect_youtube_account(
                session=session,
                label=label,
                use_device_flow=device_flow,
            )

            console.print(f"\n[bold green]Account connected successfully![/bold green]")
            console.print(f"[cyan]Account ID:[/cyan] {account.id}")
            console.print(f"[cyan]Label:[/cyan] {account.label}")
            console.print(f"[cyan]Channel:[/cyan] {account.external_name or 'Unknown'}")
            console.print(f"[cyan]Channel ID:[/cyan] {account.external_id or 'Unknown'}")

            console.print(f"\n[dim]You can now publish videos with:[/dim]")
            console.print(f"[dim]  shorts-engine shorts publish --job <uuid> --youtube-account {label}[/dim]")

    except AccountError as e:
        console.print(f"[bold red]Account error: {e}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@accounts_app.command("list")
def accounts_list(
    platform: Optional[str] = typer.Option(None, "--platform", "-p", help="Filter by platform"),
) -> None:
    """List connected platform accounts."""
    from shorts_engine.db.session import get_session_context
    from shorts_engine.services.accounts import list_accounts

    with get_session_context() as session:
        accounts = list_accounts(session, platform=platform)

        if not accounts:
            console.print("[dim]No accounts connected. Use 'shorts-engine accounts connect' to add one.[/dim]")
            return

        table = Table(title="Connected Accounts")
        table.add_column("ID", style="dim", no_wrap=True)
        table.add_column("Platform", style="cyan")
        table.add_column("Label")
        table.add_column("Channel/Account")
        table.add_column("Status", style="green")
        table.add_column("Uploads Today")
        table.add_column("Connected")

        for account in accounts:
            status_style = "green" if account.status == "active" else "red"
            table.add_row(
                str(account.id)[:8] + "...",
                account.platform,
                account.label,
                account.external_name or account.external_id or "-",
                f"[{status_style}]{account.status}[/{status_style}]",
                str(account.uploads_today or 0),
                account.created_at.strftime("%Y-%m-%d") if account.created_at else "-",
            )

        console.print(table)


@accounts_app.command("disconnect")
def accounts_disconnect(
    account_id: str = typer.Argument(..., help="Account ID or label to disconnect"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Disconnect a platform account."""
    from shorts_engine.db.models import PlatformAccountModel
    from shorts_engine.db.session import get_session_context
    from shorts_engine.services.accounts import disconnect_account, get_account_by_id, AccountNotFoundError
    from sqlalchemy import select

    with get_session_context() as session:
        # Try to find by ID first, then by label
        account = None
        try:
            account_uuid = UUID(account_id)
            account = session.get(PlatformAccountModel, account_uuid)
        except ValueError:
            # Try by label
            account = session.execute(
                select(PlatformAccountModel).where(PlatformAccountModel.label == account_id)
            ).scalar_one_or_none()

        if not account:
            console.print(f"[bold red]Account not found: {account_id}[/bold red]")
            raise typer.Exit(code=1)

        if not force:
            console.print(f"[yellow]About to disconnect:[/yellow]")
            console.print(f"  Platform: {account.platform}")
            console.print(f"  Label: {account.label}")
            console.print(f"  Channel: {account.external_name or 'Unknown'}")

            if not typer.confirm("Are you sure?"):
                console.print("[dim]Cancelled[/dim]")
                raise typer.Exit()

        disconnect_account(session, account.id)
        console.print(f"[green]Account disconnected: {account.label}[/green]")


@accounts_app.command("link")
def accounts_link(
    account_label: str = typer.Option(..., "--account", "-a", help="Account label"),
    project_id: str = typer.Option(..., "--project", "-p", help="Project ID"),
    default: bool = typer.Option(False, "--default", "-d", help="Set as default account for this project"),
) -> None:
    """Link an account to a project for publishing."""
    from shorts_engine.db.session import get_session_context
    from shorts_engine.services.accounts import (
        get_account_by_label,
        link_account_to_project,
        AccountError,
        AccountNotFoundError,
    )

    try:
        project_uuid = UUID(project_id)
    except ValueError:
        console.print(f"[bold red]Invalid project ID: {project_id}[/bold red]")
        raise typer.Exit(code=1)

    with get_session_context() as session:
        try:
            account = get_account_by_label(session, "youtube", account_label)
            link = link_account_to_project(
                session,
                account.id,
                project_uuid,
                is_default=default,
            )

            console.print(f"[green]Linked account '{account_label}' to project[/green]")
            if default:
                console.print("[dim]Set as default account for this project[/dim]")

        except (AccountNotFoundError, AccountError) as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            raise typer.Exit(code=1)


# =============================================================================
# SHORTS PUBLISH COMMAND
# =============================================================================


@shorts_app.command("publish")
def shorts_publish(
    job_id: str = typer.Option(..., "--job", "-j", help="Video job ID (UUID) to publish"),
    youtube_account: Optional[str] = typer.Option(None, "--youtube-account", "-y", help="YouTube account label"),
    publish_at: Optional[str] = typer.Option(None, "--publish-at", help="Schedule publish time (ISO 8601)"),
    visibility: str = typer.Option("public", "--visibility", "-v", help="Video visibility (public, private, unlisted)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be uploaded without actually uploading"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for publish to complete"),
) -> None:
    """Publish a rendered video to platforms.

    Requires a completed video job with a final MP4 render.

    Example:
        shorts-engine shorts publish --job <uuid> --youtube-account "Main Channel"
        shorts-engine shorts publish --job <uuid> --youtube-account "Main" --publish-at "2024-12-25T10:00:00Z"
        shorts-engine shorts publish --job <uuid> --youtube-account "Main" --dry-run
    """
    from shorts_engine.db.models import AssetModel, VideoJobModel
    from shorts_engine.db.session import get_session_context

    # Validate job ID
    try:
        job_uuid = UUID(job_id)
    except ValueError:
        console.print(f"[bold red]Invalid job ID: {job_id}[/bold red]")
        raise typer.Exit(code=1)

    # Validate visibility
    if visibility not in ("public", "private", "unlisted"):
        console.print(f"[bold red]Invalid visibility: {visibility}[/bold red]")
        console.print("[dim]Valid options: public, private, unlisted[/dim]")
        raise typer.Exit(code=1)

    # Require at least one platform
    if not youtube_account:
        console.print("[bold red]No platform specified. Use --youtube-account to specify a destination.[/bold red]")
        console.print("[dim]Example: shorts-engine shorts publish --job <uuid> --youtube-account 'Main Channel'[/dim]")
        raise typer.Exit(code=1)

    # Verify job exists and has a final video
    with get_session_context() as session:
        job = session.get(VideoJobModel, job_uuid)
        if not job:
            console.print(f"[bold red]Video job not found: {job_id}[/bold red]")
            raise typer.Exit(code=1)

        # Check for final video
        final_asset = None
        for asset in job.assets:
            if asset.asset_type == "final_video" and asset.status == "ready":
                final_asset = asset
                break

        if not final_asset:
            console.print(f"[bold red]No final video found for job.[/bold red]")
            console.print("[dim]Run 'shorts-engine shorts render --job <uuid>' first.[/dim]")
            raise typer.Exit(code=1)

        job_title = job.title or "Untitled"

    console.print(Panel.fit(
        f"[bold]Publishing Video[/bold]\n\n"
        f"[cyan]Job:[/cyan] {job_title}\n"
        f"[cyan]Job ID:[/cyan] {job_id}\n"
        f"[cyan]YouTube Account:[/cyan] {youtube_account or 'None'}\n"
        f"[cyan]Visibility:[/cyan] {visibility}\n"
        f"[cyan]Scheduled:[/cyan] {publish_at or 'Immediate'}\n"
        f"[cyan]Dry Run:[/cyan] {'Yes' if dry_run else 'No'}",
        title="Publish Pipeline",
        border_style="magenta",
    ))

    try:
        from shorts_engine.jobs.publish_pipeline import run_publish_pipeline_task

        result = run_publish_pipeline_task.delay(
            video_job_id=str(job_uuid),
            youtube_account=youtube_account,
            scheduled_publish_at=publish_at,
            visibility=visibility,
            dry_run=dry_run,
        )

        console.print(f"\n[green]Publish pipeline enqueued![/green]")
        console.print(f"[dim]Task ID: {result.id}[/dim]")

        if wait:
            console.print("\n[dim]Waiting for publish to complete...[/dim]")
            with console.status("[bold magenta]Publishing...", spinner="dots"):
                task_result = result.get(timeout=600)  # 10 min max

            if task_result.get("success"):
                console.print("\n[bold green]Published successfully![/bold green]")

                # Show platform results
                platforms = task_result.get("platforms", {})

                if "youtube" in platforms:
                    yt = platforms["youtube"]
                    if yt.get("success"):
                        if yt.get("dry_run"):
                            console.print("\n[bold cyan]YouTube (Dry Run):[/bold cyan]")
                            console.print("[dim]Would have uploaded with the following settings:[/dim]")
                            payload = yt.get("payload", {})
                            console.print(f"  Title: {payload.get('metadata', {}).get('snippet', {}).get('title', 'N/A')}")
                        else:
                            console.print(f"\n[bold cyan]YouTube:[/bold cyan]")
                            console.print(f"  URL: {yt.get('url')}")
                            console.print(f"  Video ID: {yt.get('platform_video_id')}")
                            if yt.get("forced_private"):
                                console.print("  [yellow]Note: Video was forced to private by YouTube API[/yellow]")
                    else:
                        console.print(f"\n[bold red]YouTube failed:[/bold red] {yt.get('error')}")
            else:
                console.print(f"\n[bold red]Publish failed![/bold red]")
                for platform, result_data in task_result.get("platforms", {}).items():
                    if not result_data.get("success"):
                        console.print(f"  {platform}: {result_data.get('error')}")
                raise typer.Exit(code=1)
        else:
            console.print(f"\n[dim]Use 'shorts-engine shorts status {result.id}' to check progress[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


# =============================================================================
# INGEST COMMANDS
# =============================================================================


def _parse_duration(duration: str) -> int:
    """Parse a duration string to hours.

    Supports formats like: 24h, 7d, 1w, 168

    Args:
        duration: Duration string (e.g., "24h", "7d", "1w").

    Returns:
        Number of hours.
    """
    duration = duration.lower().strip()

    if duration.endswith("h"):
        return int(duration[:-1])
    elif duration.endswith("d"):
        return int(duration[:-1]) * 24
    elif duration.endswith("w"):
        return int(duration[:-1]) * 24 * 7
    else:
        # Assume hours if no suffix
        return int(duration)


@ingest_app.command("metrics")
def ingest_metrics(
    since: str = typer.Option("24h", "--since", "-s", help="Time window (e.g., 24h, 7d, 1w)"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for completion"),
) -> None:
    """Ingest performance metrics for published videos.

    Fetches metrics from YouTube Analytics API for all videos published
    within the specified time window. Metrics are stored with time-windowed
    snapshots (1h, 6h, 24h, 72h, 7d) and reward scores are computed.

    Example:
        shorts-engine ingest metrics --since 7d
        shorts-engine ingest metrics --since 24h --wait
    """
    try:
        since_hours = _parse_duration(since)
    except ValueError:
        console.print(f"[bold red]Invalid duration format: {since}[/bold red]")
        console.print("[dim]Use formats like: 24h, 7d, 1w[/dim]")
        raise typer.Exit(code=1)

    console.print(f"[bold blue]Ingesting metrics for videos published in the last {since}...[/bold blue]")

    try:
        from shorts_engine.jobs.ingestion_tasks import ingest_metrics_batch_task

        result = ingest_metrics_batch_task.delay(since_hours=since_hours)
        console.print(f"[dim]Task ID: {result.id}[/dim]")

        if wait:
            console.print("[dim]Waiting for completion...[/dim]")
            with console.status("[bold blue]Ingesting metrics...", spinner="dots"):
                task_result = result.get(timeout=3600)

            if task_result.get("success"):
                console.print(f"[bold green]Ingestion complete![/bold green]")

                table = Table(title="Metrics Ingestion Result")
                table.add_column("Metric", style="cyan")
                table.add_column("Value")

                table.add_row("Videos Processed", str(task_result.get("processed", 0)))
                table.add_row("Metrics Created/Updated", str(task_result.get("metrics_created", 0)))
                table.add_row("Errors", str(task_result.get("errors", 0)))

                console.print(table)
            else:
                console.print(f"[bold red]Ingestion failed[/bold red]")
                console.print(f"[red]{task_result.get('error', 'Unknown error')}[/red]")
                raise typer.Exit(code=1)
        else:
            console.print(f"[green]Task enqueued successfully![/green]")
            console.print(f"[dim]Use 'shorts-engine status {result.id}' to check progress[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@ingest_app.command("comments")
def ingest_comments(
    since: str = typer.Option("24h", "--since", "-s", help="Time window (e.g., 24h, 7d, 1w)"),
    max_per_video: int = typer.Option(100, "--max", "-m", help="Max comments per video"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for completion"),
) -> None:
    """Ingest comments for published videos.

    Fetches top-level comments from YouTube Data API for all videos published
    within the specified time window. Comments are stored for sentiment analysis
    and learning.

    Example:
        shorts-engine ingest comments --since 7d
        shorts-engine ingest comments --since 24h --max 50 --wait
    """
    try:
        since_hours = _parse_duration(since)
    except ValueError:
        console.print(f"[bold red]Invalid duration format: {since}[/bold red]")
        console.print("[dim]Use formats like: 24h, 7d, 1w[/dim]")
        raise typer.Exit(code=1)

    console.print(f"[bold blue]Ingesting comments for videos published in the last {since}...[/bold blue]")
    console.print(f"[dim]Max {max_per_video} comments per video[/dim]")

    try:
        from shorts_engine.jobs.ingestion_tasks import ingest_comments_batch_task

        result = ingest_comments_batch_task.delay(
            since_hours=since_hours,
            max_per_video=max_per_video,
        )
        console.print(f"[dim]Task ID: {result.id}[/dim]")

        if wait:
            console.print("[dim]Waiting for completion...[/dim]")
            with console.status("[bold blue]Ingesting comments...", spinner="dots"):
                task_result = result.get(timeout=3600)

            if task_result.get("success"):
                console.print(f"[bold green]Ingestion complete![/bold green]")

                table = Table(title="Comments Ingestion Result")
                table.add_column("Metric", style="cyan")
                table.add_column("Value")

                table.add_row("Videos Processed", str(task_result.get("processed", 0)))
                table.add_row("Total Comments", str(task_result.get("total_comments", 0)))
                table.add_row("Errors", str(task_result.get("errors", 0)))

                console.print(table)
            else:
                console.print(f"[bold red]Ingestion failed[/bold red]")
                console.print(f"[red]{task_result.get('error', 'Unknown error')}[/red]")
                raise typer.Exit(code=1)
        else:
            console.print(f"[green]Task enqueued successfully![/green]")
            console.print(f"[dim]Use 'shorts-engine status {result.id}' to check progress[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
