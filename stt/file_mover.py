"""
File Mover Module
Handles automatic file moving to remote locations (SMB/NAS) with retry logic
"""

import os
import sys
import shutil
import glob
import logging
import platform

logger = logging.getLogger(__name__)

# SMB/CIFS direct access (no mounting required)
try:
    from smbclient import (register_session, open_file, mkdir, remove as smb_remove,
                           stat as smb_stat)
    from smbclient.path import exists as smb_exists
    SMB_AVAILABLE = True
except ImportError:
    SMB_AVAILABLE = False
    logger.warning("smbprotocol not installed - SMB direct access unavailable")

def is_smb_path(path):
    """Check if path is an SMB/network path"""
    return path.startswith('//') or path.startswith('\\\\')

def copy_file_to_smb_direct(local_file, smb_dest_path, username, password, domain=''):
    """
    Copy file directly to SMB share without mounting
    Uses smbprotocol for direct network file operations

    Args:
        local_file: Local file path to copy
        smb_dest_path: SMB destination path (format: //server/share/folder/file.ext)
        username: SMB username
        password: SMB password
        domain: SMB domain (optional)

    Returns:
        (success: bool, error: str or None)
    """
    if not SMB_AVAILABLE:
        return False, "smbprotocol library not installed"

    try:
        # Normalize path separators to forward slashes
        normalized = smb_dest_path.replace('\\', '/')

        # Parse SMB path to get server
        parts = normalized.replace('//', '').split('/')
        if len(parts) < 2:
            return False, f"Invalid SMB path format: {smb_dest_path}"

        server = parts[0]
        logger.info(f"SMB direct copy: {local_file} -> {smb_dest_path} (server={server}, user={username})")

        # Register SMB session with credentials
        register_session(server, username=username, password=password, auth_protocol='ntlm')

        # Ensure parent directory exists
        dest_dir = '/'.join(normalized.split('/')[:-1])  # Remove filename
        if dest_dir and not smb_exists(dest_dir):
            # Create directory recursively
            # Start from //server/share (need at least 2 parts)
            parts_list = dest_dir.replace('//', '').split('/')

            for i, part in enumerate(parts_list):
                # Build path incrementally
                if i == 0:
                    # First part is server: //10.1.10.6
                    current_path = '//' + part
                elif i == 1:
                    # Second part is share: //10.1.10.6/tt
                    current_path = '//' + parts_list[0] + '/' + part
                else:
                    # Subsequent parts are directories
                    current_path += '/' + part

                # Skip server-only paths (need at least server + share)
                if i == 0:
                    continue

                if not smb_exists(current_path):
                    try:
                        logger.info(f"SMB creating dir: {current_path}")
                        mkdir(current_path)
                    except Exception as mkdir_err:
                        logger.warning(f"SMB mkdir failed for {current_path}: {mkdir_err}")

        # Copy file to SMB share
        logger.info(f"SMB opening file for write: {normalized}")
        with open(local_file, 'rb') as src:
            with open_file(normalized, mode='wb') as dst:
                shutil.copyfileobj(src, dst)

        logger.info(f"Successfully copied to SMB: {smb_dest_path}")
        return True, None

    except Exception as e:
        error_msg = f"SMB copy failed: {e}"
        logger.error(error_msg)
        return False, str(e)

def mount_smb_share(smb_path, username, password, domain=''):
    """
    Mount SMB share (cross-platform)
    Windows: uses net use command
    Linux: uses mount.cifs
    Returns True if successful, False otherwise
    """
    import subprocess

    try:
        system = platform.system()

        if system == 'Windows':
            # Parse SMB path
            if smb_path.startswith('//'):
                smb_path = smb_path.replace('/', '\\')

            # Build credential string
            user_str = f'{domain}\\{username}' if domain else username

            # Try to mount the share. NOTE: the password appears in the process
            # command line for the duration of the call; Windows offers no
            # non-interactive way to pass it that avoids this (cmdkey has the
            # same exposure), and unlike /proc on Linux it's only visible to
            # administrators.
            cmd = ['net', 'use', smb_path, f'/user:{user_str}', password]
            # creationflags: windowless server — 'net use' would flash a console
            # window on every (re)mount.
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

            if result.returncode == 0 or 'already in use' in result.stdout.lower():
                logger.info(f"SMB share mounted successfully: {smb_path}")
                return True
            else:
                logger.error(f"Failed to mount SMB share: {result.stderr}")
                return False

        elif system == 'Linux':
            # Convert Windows path format to UNC format
            unc_path = smb_path.replace('\\', '/')
            if not unc_path.startswith('//'):
                unc_path = '//' + unc_path

            # Extract share and server
            parts = unc_path.replace('//', '').split('/')
            if len(parts) < 2:
                logger.error(f"Invalid SMB path format: {smb_path}")
                return False

            server = parts[0]
            share = parts[1]

            # Create mount point in /mnt if it doesn't exist
            mount_point = f'/mnt/{server}_{share}'
            os.makedirs(mount_point, exist_ok=True)

            # Check if already mounted
            with open('/proc/mounts', 'r') as f:
                if mount_point in f.read():
                    logger.info(f"SMB share already mounted at {mount_point}")
                    return True

            # Build mount command
            cmd = ['sudo', 'mount', '-t', 'cifs', f'//{server}/{share}', mount_point]

            # Pass credentials via a 0600 temp file instead of -o password=...,
            # which would be world-readable in /proc/<pid>/cmdline for the
            # duration of the mount call
            import tempfile
            creds_file = None
            try:
                if username or password or domain:
                    fd, creds_file = tempfile.mkstemp(prefix='.smbcreds-')
                    with os.fdopen(fd, 'w') as f:
                        if username:
                            f.write(f'username={username}\n')
                        if password:
                            f.write(f'password={password}\n')
                        if domain:
                            f.write(f'domain={domain}\n')
                    cmd.extend(['-o', f'credentials={creds_file}'])

                # Try to mount
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            finally:
                if creds_file:
                    try:
                        os.unlink(creds_file)
                    except OSError:
                        pass

            if result.returncode == 0:
                logger.info(f"SMB share mounted successfully at {mount_point}")
                return True
            else:
                logger.error(f"Failed to mount SMB share: {result.stderr}")
                return False
        else:
            logger.error(f"Unsupported platform for SMB mounting: {system}")
            return False

    except Exception as e:
        logger.error(f"Error mounting SMB share: {e}")
        return False

def test_destination_accessible(dest_path, username='', password='', domain=''):
    """
    Test if destination path is accessible
    For SMB paths, uses direct SMB access (no mounting required)
    For local paths, uses standard filesystem operations
    """
    try:
        if is_smb_path(dest_path):
            # Use direct SMB access
            if not SMB_AVAILABLE:
                logger.error("smbprotocol library not installed")
                return False

            # Normalize path separators
            normalized = dest_path.replace('\\', '/')
            parts = normalized.replace('//', '').split('/')

            if len(parts) < 2:
                logger.error(f"Invalid SMB path format (needs server/share): {dest_path}")
                return False

            server = parts[0]

            # Register SMB session with credentials
            if username and password:
                register_session(server, username=username, password=password, auth_protocol='ntlm')
            else:
                logger.error("SMB credentials required")
                return False

            # Test write access by creating a test file
            test_file_path = normalized.rstrip('/') + '/.file_mover_test'

            try:
                with open_file(test_file_path, mode='w') as f:
                    f.write('test')

                # Clean up test file
                smb_remove(test_file_path)

                logger.info(f"SMB destination accessible: {dest_path}")
                return True

            except Exception as e:
                logger.error(f"SMB test write failed: {e}")
                return False

        else:
            # Local path - use standard filesystem operations
            os.makedirs(dest_path, exist_ok=True)

            # Test write access with a temporary file
            test_file = os.path.join(dest_path, '.file_mover_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)

            logger.info(f"Local destination accessible: {dest_path}")
            return True

    except Exception as e:
        logger.error(f"Destination not accessible: {e}")
        return False

# Returned in the error slot of a successful call to say "nothing was transferred".
# A sentinel rather than a third return value, so every existing caller that only
# looks at the success flag keeps working.
SKIPPED_ALREADY_PRESENT = 'skipped:already-present'


def _destination_matches(source_file, dest_file, username='', password='', domain=''):
    """Whether ``dest_file`` already holds a file of the same size as the source.

    Any doubt answers False — a re-copy costs bandwidth, a wrong skip loses a
    recording, and only one of those is recoverable. So an unreadable destination, a
    missing SMB library, or a stat that raises all mean "copy it".
    """
    try:
        source_size = os.path.getsize(source_file)
    except OSError:
        return False

    if is_smb_path(dest_file):
        if not SMB_AVAILABLE:
            return False
        try:
            normalized = dest_file.replace('\\', '/')
            if not smb_exists(normalized):
                return False
            return smb_stat(normalized).st_size == source_size
        except Exception:
            return False

    try:
        return os.path.isfile(dest_file) and os.path.getsize(dest_file) == source_size
    except OSError:
        return False


def move_file_with_structure(source_file, dest_base, working_dir, preserve_structure=True, delete_source=True, username='', password='', domain=''):
    """
    Move a file to destination, optionally preserving directory structure
    Supports both local paths and SMB shares (direct access without mounting)
    """
    try:
        # Get relative path from working directory
        rel_path = os.path.relpath(source_file, working_dir)

        if preserve_structure:
            # Preserve the directory structure
            if is_smb_path(dest_base):
                # For SMB paths, manually join with forward slashes (don't use os.path.join)
                dest_file = dest_base.rstrip('/') + '/' + rel_path.replace('\\', '/')
            else:
                # For local paths, use os.path.join
                dest_file = os.path.join(dest_base, rel_path)
        else:
            # Just use the filename
            filename = os.path.basename(source_file)
            if is_smb_path(dest_base):
                # For SMB paths, manually join with forward slashes
                dest_file = dest_base.rstrip('/') + '/' + filename
            else:
                # For local paths, use os.path.join
                dest_file = os.path.join(dest_base, filename)

        # In copy mode, skip what is already there at the same size.
        #
        # find_files_to_move globs a pattern rather than tracking what is new, and
        # deletion is what normally keeps that honest — so with delete_source off,
        # every run re-sent the entire accumulated history and each service took
        # longer than the last. Size is enough to tell a delivered file from a new
        # one here (these are finished recordings, never rewritten in place) and it
        # needs no state file, which also makes an interrupted run cheaply resumable.
        if not delete_source and _destination_matches(source_file, dest_file,
                                                      username, password, domain):
            logger.info(f"Already present, skipping: {dest_file}")
            return True, SKIPPED_ALREADY_PRESENT

        # Handle SMB vs local paths
        if is_smb_path(dest_base):
            # Direct SMB copy (no mounting required)
            success, error = copy_file_to_smb_direct(source_file, dest_file, username, password, domain)

            if not success:
                raise Exception(error)

            # Delete source if requested
            if delete_source:
                os.remove(source_file)
                logger.info(f"Moved: {source_file} -> {dest_file}")
            else:
                logger.info(f"Copied: {source_file} -> {dest_file}")

        else:
            # Local file operations
            dest_dir = os.path.dirname(dest_file)
            os.makedirs(dest_dir, exist_ok=True)

            # Copy or move the file
            if delete_source:
                shutil.move(source_file, dest_file)
                logger.info(f"Moved: {source_file} -> {dest_file}")
            else:
                shutil.copy2(source_file, dest_file)
                logger.info(f"Copied: {source_file} -> {dest_file}")

        return True, None

    except Exception as e:
        error_msg = f"Failed to move {source_file}: {e}"
        logger.error(error_msg)
        return False, str(e)

def get_base_directories_from_patterns(patterns, working_dir):
    """
    Extract base directories from source patterns

    For patterns like ['_AUTOMATIC_BACKUP/**/*.wav', 'logs/**/*.txt'],
    returns set(['/absolute/path/_AUTOMATIC_BACKUP', '/absolute/path/logs'])

    Returns a set of absolute paths to base directories
    """
    base_dirs = set()

    for pattern in patterns:
        # Remove glob characters to find the base directory
        parts = pattern.split('/')
        base_parts = []

        for part in parts:
            if '*' in part or '?' in part or '[' in part:
                # Stop at first glob pattern
                break
            base_parts.append(part)

        if base_parts:
            base_dir = os.path.join(working_dir, *base_parts)
            base_dirs.add(os.path.abspath(base_dir))

    # If no base directories found, use working_dir
    if not base_dirs:
        base_dirs.add(os.path.abspath(working_dir))

    return base_dirs

def get_base_directory_for_file(file_path, base_dirs):
    """
    Find which base directory a file belongs to

    Args:
        file_path: Absolute path to the file
        base_dirs: Set of absolute paths to base directories

    Returns:
        The base directory that this file is under, or None
    """
    file_path = os.path.abspath(file_path)

    # Find the base directory that this file is under
    for base_dir in base_dirs:
        if file_path.startswith(base_dir + os.sep):
            return base_dir

    return None

def cleanup_empty_directories(file_path, base_dir):
    """
    Remove empty parent directories after moving a file, stopping at base_dir

    Args:
        file_path: Path of the file that was moved
        base_dir: Base directory to preserve (don't delete this)
    """
    try:
        # Get the parent directory of the file
        parent_dir = os.path.dirname(file_path)

        # Normalize base_dir to absolute path
        base_dir = os.path.abspath(base_dir)

        # Walk up the directory tree
        while True:
            parent_dir = os.path.abspath(parent_dir)

            # Stop if we've reached the base directory
            if parent_dir == base_dir or not parent_dir.startswith(base_dir):
                break

            # Check if directory is empty
            try:
                if not os.listdir(parent_dir):
                    logger.info(f"Removing empty directory: {parent_dir}")
                    os.rmdir(parent_dir)
                    # Move up to parent
                    parent_dir = os.path.dirname(parent_dir)
                else:
                    # Directory not empty, stop here
                    break
            except OSError as e:
                # Can't remove directory (permissions, not empty, etc.)
                logger.debug(f"Could not remove directory {parent_dir}: {e}")
                break

    except Exception as e:
        logger.warning(f"Error cleaning up empty directories for {file_path}: {e}")

def find_files_to_move(patterns, working_dir):
    """
    Find files matching patterns
    """
    files_to_move = []

    for pattern in patterns:
        full_pattern = os.path.join(working_dir, pattern)

        for file_path in glob.glob(full_pattern, recursive=True):
            if os.path.isfile(file_path):
                files_to_move.append(file_path)

    return files_to_move

def execute_file_move(config_getter, working_dir=None):
    """
    Execute file moving operation synchronously

    This function contains the core logic for finding and moving files.
    It can be called directly from anywhere that needs to trigger file moving.

    Args:
        config_getter: Function that returns the current config dict
        working_dir: Directory the source patterns are relative to — the data
            directory holding _AUTOMATIC_BACKUP. Callers must pass it; the
            fallback below only exists for older callers and cannot be trusted.

    Returns:
        dict: {
            'success': bool,
            'moved': int,
            'failed': int,
            'errors': list of error messages,
            'message': str summary
        }
    """
    errors = []

    try:
        config = config_getter()
        mover_config = config.get('file_manager', {}).get('file_mover', {})

        dest_path = mover_config.get('destination_path', '').strip()
        if not dest_path:
            return {
                'success': False,
                'moved': 0,
                'failed': 0,
                'errors': ['No destination path configured'],
                'message': 'No destination path configured'
            }

        print(f"[EXECUTE] Destination: {dest_path}", flush=True)

        # Get settings
        username = mover_config.get('smb_username', '')
        password = mover_config.get('smb_password', '')
        domain = mover_config.get('smb_domain', '')
        patterns = mover_config.get('source_patterns', [])
        delete_source = mover_config.get('delete_source', True)
        preserve_structure = mover_config.get('preserve_structure', True)
        # Default false: turning this on changes where files land, and an existing
        # archive's layout must not shift under it because the app updated.
        strip_source_base = mover_config.get('strip_source_base', False)

        # Source patterns are relative to the data directory, which the caller
        # knows and this module must not guess. It used to reach into
        # sys.modules['speech_to_text'].APP_DIR — but the server runs the
        # monolith as a script, so that module is registered as __main__ and the
        # lookup always returned None. The fallback then pointed at stt/ itself,
        # where no backup has ever lived, and every run reported "Moved 0 files"
        # against a destination it had just successfully reached.
        if not working_dir:
            main_mod = sys.modules.get('speech_to_text') or sys.modules.get('__main__')
            working_dir = (getattr(main_mod, 'APP_DIR', None)
                           or os.path.dirname(os.path.abspath(__file__)))
            print(f"[EXECUTE] No working_dir passed; guessed {working_dir}", flush=True)
        base_dirs = get_base_directories_from_patterns(patterns, working_dir)

        # Reachability is a warning, not a verdict. The probe writes at the top of
        # dest_path, but every file lands in a subdirectory of it — and a NAS that
        # grants write inside a folder while denying it at the share root is a common
        # layout, not a misconfiguration. Aborting there moved nothing at all, and
        # with delete_source on, sessions simply accumulated locally. The per-file
        # copies below report their own failures, so let them.
        print("[EXECUTE] Testing destination accessibility...", flush=True)
        if test_destination_accessible(dest_path, username, password, domain):
            print("[EXECUTE] Destination accessible", flush=True)
        else:
            warning = (f'Destination root not writable: {dest_path} — continuing, since '
                       f'files are written to subdirectories of it')
            print(f"[EXECUTE] {warning}", flush=True)
            logger.warning(warning)
            errors.append(warning)

        # Find files to move
        files_to_move = find_files_to_move(patterns, working_dir)
        print(f"[EXECUTE] Found {len(files_to_move)} files to move", flush=True)

        # Move files
        moved_count = 0
        failed_count = 0
        skipped_count = 0

        for file_path in files_to_move:
            if not os.path.exists(file_path):
                continue

            print(f"[EXECUTE] Processing: {file_path}", flush=True)
            # With strip_source_base the path is taken relative to the pattern's own
            # base directory, so a destination named after that directory does not
            # end up containing a second copy of it.
            file_base = (get_base_directory_for_file(file_path, base_dirs) or working_dir
                         ) if strip_source_base else working_dir
            success, error = move_file_with_structure(
                file_path, dest_path, file_base,
                preserve_structure, delete_source,
                username, password, domain
            )

            if error == SKIPPED_ALREADY_PRESENT:
                skipped_count += 1
                continue

            if success:
                moved_count += 1
                # Clean up empty directories if we deleted the source
                if delete_source:
                    # Find which base directory this file belongs to
                    base_dir = get_base_directory_for_file(file_path, base_dirs)
                    if base_dir:
                        cleanup_empty_directories(file_path, base_dir)
            else:
                failed_count += 1
                errors.append(f"{file_path}: {error}")

        message = (f"Moved {moved_count} files"
                   + (f", {skipped_count} already present" if skipped_count > 0 else "")
                   + (f", {failed_count} failed" if failed_count > 0 else ""))
        print(f"[EXECUTE] Complete: {message}", flush=True)

        return {
            'success': True,
            'moved': moved_count,
            'skipped': skipped_count,
            'failed': failed_count,
            'errors': errors,
            'message': message,
            'delete_source': delete_source
        }

    except Exception as e:
        error_msg = f"Error executing file move: {e}"
        logger.error(error_msg)
        print(f"[EXECUTE] ERROR: {error_msg}", flush=True)
        return {
            'success': False,
            'moved': 0,
            'failed': 0,
            'errors': [str(e)],
            'message': error_msg
        }

def execute_file_move_now(config_getter, working_dir=None):
    """
    Execute file moving immediately in the current thread

    This is the direct execution path for transcription stop.
    Unlike trigger_file_mover(), this executes synchronously instead
    of signaling the worker thread.

    Args:
        config_getter: Function that returns the current config dict
        working_dir: Directory the source patterns are relative to

    Returns:
        dict: Result from execute_file_move()
    """
    logger.info("Direct file move execution requested")
    print("[FILE MOVER] Executing file move directly...", flush=True)

    result = execute_file_move(config_getter, working_dir)

    if result['success']:
        print(f"[FILE MOVER] ✓ Complete: Moved {result['moved']}, Failed {result['failed']}", flush=True)
    else:
        print(f"[FILE MOVER] ✗ Error: {result.get('message', 'Unknown error')}", flush=True)

    return result
